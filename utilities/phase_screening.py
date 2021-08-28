#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2021/1/29 9:42
# @author : Xujun Zhang

import os
import time
import pandas as pd
import numpy as np
# from multiprocessing import Pool
from pathos.multiprocessing import ProcessPool as Pool
from my_utils import Smiles, Pharmacophore, get_label
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # init
    path = r'/home/xujun/Project_5/pharmacophore'  # ./
    path_generated = f'{path}/generated'
    data_path = f'{path}/src'
    result_path = f'{path}/dst'
    model_type = 'pcb'  # fpb pcb
    smile_txt = f'{data_path}/{model_type}_test.txt'
    similar_smile_csv = f'{result_path}/{model_type}_similar.csv'
    pharmer = Pharmacophore()
    if not os.path.exists(similar_smile_csv):
        pd.DataFrame(['name', 'smile', 'mw', 'logp', 'rb', 'hba', 'hbr', 'halx', 'similarity', 'label']).T.to_csv(similar_smile_csv, index=False, header=None)
    # get smiles
    with open(smile_txt, 'r') as f:
        contents = f.read().split('<EOS>\n')[:-1]  # return list
    # for each seed smile
    for seed, content in enumerate(contents[:]):
        # seed = seed + 1297
        content = content.splitlines()
        # collate
        smiles, mw, logp, rb, hba, hbr, halx = [], [], [], [], [], [], []
        names, similarities = [], []
        # each generated smile
        for smi_num, line in enumerate(content[:]):
            line = line.split()
            names.append(f'{seed}_{smi_num}')
            similarities.append(float(line[7]))
            for i, lis in enumerate([smiles, mw, logp, rb, hba, hbr, halx]):
                lis.append(line[i])
        # trans to df
        df = pd.DataFrame([names, smiles, mw, logp, rb, hba, hbr, halx, similarities]).T
        if model_type == 'pcb':
            df_seed = pd.DataFrame(df.iloc[0, :]).T  # get seed
            df = df[df.iloc[:, -1] <= 0.4]  # [:, -1] >= 0.7] [:, -1] <= 0.41]
            df = df_seed.append(df, sort=False)
            # add label
            df_ac = df[df.iloc[:, 0].str.contains('_0')]
            df_ac['label'] = np.ones((len(df_ac)))
            df_inac = df[~df.iloc[:, 0].str.contains('_0')]
            df_inac['label'] = np.zeros((len(df_inac)))
            # merge
            df = df_ac.append(df_inac, sort=False)
        else:
            # filter for similarity
            df = df[df.iloc[:, -1] >= 0.7]
            # update names and smiles
            names, smiles = df.iloc[:, 0].tolist(), df.iloc[:, 1].tolist()
            # trans smiles to sdf
            path_generated_local = f'{path_generated}/{seed}'
            if not os.path.exists(path_generated_local):
                os.mkdir(path_generated_local)
            #
            smiler = Smiles(smile_lis=smiles, names=names, save_path=path_generated_local)
            # multi process
            pool = Pool(28)
            print(f'transform to 3D... total:{len(smiler.mols)}')
            mols = pool.map(smiler.to3D, smiler.mols)
            print('write to sdf file...')
            # pool.starmap(smiler.save_to_file, zip(mols, smiler.names))
            valid_names = pool.map(smiler.save_to_file, mols, smiler.names)
            # filter None
            valid_names = list(filter(lambda x: x is not None, valid_names))
            print('merge file for screening')
            pool.clear()
            # pool.close()
            # pool.join()
            unscreened_ligand = f'{path_generated_local}/unscreenedLigand'
            smiler.merge_file(src_files=valid_names, dst_file=unscreened_ligand, format='sdf')
            # rename file before delete
            src_seed_ligand = f'{path_generated_local}/{seed}_0'
            dst_seed_ligand = f'{path_generated_local}/seed'
            os.rename(src_seed_ligand+'.sdf', dst_seed_ligand+'.sdf')
            print('rmove raw sdf files....')
            os.system(f'rm {path_generated_local}/*_*.sdf')
            # pharmacophore
            hyper_prefix = '0'
            screened_report = f'{path_generated_local}/{hyper_prefix}_1.rpt'
            print('start screening....')
            pharmer.phase_screen_file(path_generated_local, unscreened_ligand+'.sdf', dst_seed_ligand+'.sdf', hyper_prefix)
            # print('write to result file')
            # append2txt(result_txt, screened_report)
            print('add label and write to csv')
            labels = get_label(screened_report, valid_names)
            if labels is not None:
                df.index = df.iloc[:, 0]
                df = df.loc[valid_names, :]
                df['label'] = labels
                # rm
                os.system(f'rm -rf {path_generated_local}')
        # add to csv
        df.to_csv(similar_smile_csv, index=False, header=None, mode='a')


