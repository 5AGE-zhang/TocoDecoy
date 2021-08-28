#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2021/4/12 19:27
# @author : Xujun Zhang
import os
import shutil
from functools import reduce, partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE


# def
def grid_sample(df_tmp, grid_2, grid_unit):
    tmp_result = []
    for j in range(grid_unit):
        df_tmp_ = df_tmp[(df_tmp.tsne_2 >= grid_2[j]) & (df_tmp.tsne_2 < grid_2[j + 1])]
        idx = df_tmp_.index.values
        if idx.any():
            median_idx = idx[idx.shape[0] // 2]
            tmp_result.append(df_tmp_.loc[median_idx, :])
    return tmp_result


class grid_filter():

    def __init__(self, src_csv, ecfp_csv, tsne_csv, dst_csv, decoys_size=50, grid_unit=None):
        df = pd.read_csv(src_csv)
        self.src_csv = src_csv
        self.smiles = df.smile.values
        self.names = df.name.values
        self.trains = df.train.values
        self.labels = df.label.values
        del df
        self.ecfp_csv = ecfp_csv
        self.tsne_csv = tsne_csv
        self.decoys_size = decoys_size
        self.dst_csv = dst_csv
        # ecfp
        if not os.path.exists(ecfp_csv):
            self.bad_idxs = []
            print('cal ecfp')
            pd.DataFrame(['name'] + [i for i in range(2048)] + ['label', 'train']).T.to_csv(self.ecfp_csv, encoding='utf-8',
                                                                                   index=False, header=None)
            pool = Pool(50)
            self.bad_idxs = pool.map(self.cal_ecfp_write, range(len(self.names)))
            pool.close()
            pool.join()
            # drop bad term
            self.bad_idxs = sorted([i for i in self.bad_idxs if i], reverse=True)
            self.names = np.delete(self.names, self.bad_idxs)
            self.smiles = np.delete(self.smiles, self.bad_idxs)
            self.labels = np.delete(self.labels, self.bad_idxs)
        # tsne
        if not os.path.exists(self.tsne_csv):
            print('cal tsne')
            self.cal_tnse()
        # grid cluster
        if not os.path.exists(self.dst_csv):
            print('grid filtering')
            self.grid_cluster(grid_unit)

    def cal_ecfp_write(self, idx):
        lig_name, lig_smile, lig_label, lig_train = self.names[idx], self.smiles[idx], self.labels[idx], self.trains[idx]
        tmp_mol = Chem.MolFromSmiles(lig_smile)
        try:
            ecfp = AllChem.GetMorganFingerprintAsBitVect(tmp_mol, 2, 2048).ToBitString()
            tmp = [lig_name] + [i for i in ecfp] + [lig_label] + [lig_train]
            pd.DataFrame(tmp).T.to_csv(self.ecfp_csv, encoding='utf-8', index=False, header=None, mode='a')
            return None
        except:
            return idx

    def cal_tnse(self):
        tmp_df = pd.read_csv(self.ecfp_csv)
        names = tmp_df.iloc[:, 0]
        ecfps = tmp_df.iloc[:, 1:-2].values
        labels = tmp_df.iloc[:, -2]
        trains = tmp_df.iloc[:, -1]
        # tsne
        t_sne = TSNE(n_components=2, n_jobs=-1, random_state=42, perplexity=100, learning_rate=1000, n_iter=2000)
        y = t_sne.fit_transform(ecfps)
        #
        sne = pd.DataFrame(y, columns=['tsne_1', 'tsne_2'])
        df_a_sne = pd.concat([names, sne, labels, trains], axis=1)
        df_a_sne.to_csv(self.tsne_csv, index=False)

    def cluster_(self, tsnes, df_decoys, grid_unit):
        min_1, min_2 = tsnes.min(axis=0)
        max_1, max_2 = tsnes.max(axis=0)
        grid_1 = np.linspace(min_1, max_1, grid_unit + 1)
        grid_2 = np.linspace(min_2, max_2, grid_unit + 1)
        # split to df_
        print('split df')
        df_tmps = [df_decoys[(df_decoys.tsne_1 >= grid_1[i]) & (df_decoys.tsne_1 < grid_1[i + 1])] for i in
                   range(grid_unit)]

        grid_sample_ = partial(grid_sample, grid_2=grid_2, grid_unit=grid_unit)
        # multi processing
        print('start grid sample')
        pool = Pool(50)
        df_result = pool.map(grid_sample_, df_tmps)
        pool.close()
        pool.join()
        # filter none
        print('filtering')
        df_result = list(filter(lambda x: x != [], df_result))

        # extend
        def extendd(lis1, lis2):
            lis1.extend(lis2)
            return lis1

        df_result = list(reduce(extendd, df_result))
        # concat
        df_decoys = pd.concat(df_result, axis=1).T
        return df_decoys

    def grid_cluster(self, grid_unit):
        tmp_df = pd.read_csv(self.tsne_csv)
        df_ac = tmp_df[tmp_df.label == 1]
        df_decoys = tmp_df[tmp_df.label == 0]
        target_num = self.decoys_size * len(df_ac)
        if len(df_decoys) < target_num:
            print('no need grid filtering')
            shutil.copy(src=self.src_csv, dst=self.dst_csv)
        else:
            tsnes = df_decoys.iloc[:, 1:-2].values
            if isinstance(grid_unit, int):
                df_decoys = self.cluster_(tsnes, df_decoys, grid_unit)
                idx_lens = len(df_decoys)
                print(f'grid_unit:{grid_unit} || decoys_num:{idx_lens}')
            else:
                idx_lens = len(tsnes)
                grid_unit = 1050
                while not (target_num - 100 < idx_lens < target_num + 100):
                # while grid_unit > 1000:
                    grid_unit -= 10
                    df_decoys = self.cluster_(tsnes, df_decoys, grid_unit)
                    #
                    idx_lens = len(df_decoys)
                    print(f'grid_unit:{grid_unit} || decoys_num:{idx_lens}')
            # end
            # df_decoys.to_csv(self.tsne_csv, index=False)
            # merge
            df_src = pd.read_csv(self.src_csv)
            src_ac = df_src[df_src.label == 1]
            src_decoys = df_src[df_src.label == 0]
            dst_ac = pd.merge(df_ac.iloc[:, :-2], src_ac, on='name', how='left')
            print(dst_ac)
            dst_decoys = pd.merge(df_decoys.iloc[:, :-2], src_decoys, on='name', how='left')
            print(dst_decoys)
            dst_df = dst_ac.append(dst_decoys, sort=False)
            print(dst_df)
            dst_df.to_csv(self.dst_csv, index=False)
