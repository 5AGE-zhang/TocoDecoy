#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2021/4/12 18:37
# @author : Xujun Zhang

import argparse
import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from functools import reduce
from class_base.property_filter_base import properties_filer

warnings.filterwarnings('ignore')


def merge_2_df(df1, df2):
    new_df = df1.append(df2, sort=False)
    return new_df


if __name__ == '__main__':
    # init
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--path', type=str, default='/root/zxj/du/total_workflow')
    argparser.add_argument('--target', type=str, default='du')
    argparser.add_argument('--src_path', type=str, default='2_propterties_filter')
    argparser.add_argument('--dst_path', type=str, default='2_propterties_filter')
    argparser.add_argument('--src_file', type=str, default='dst_smi.txt')
    argparser.add_argument('--tmp_file', type=str, default='property_unfiltered.csv')
    argparser.add_argument('--dst_file', type=str, default='property_filtered.csv')
    args = argparser.parse_args()
    #
    path = f'{args.path}/{args.target}'
    src_path = f'{path}/{args.src_path}'
    dst_path = f'{path}/{args.dst_path}'
    src_txt = f'{src_path}/{args.src_file}'
    tmp_csv = f'{dst_path}/{args.tmp_file}'
    dst_csv = f'{dst_path}/{args.dst_file}'
    if not os.path.exists(tmp_csv):
        print('collate data from txt to csv....')
        pd.DataFrame(['name', 'smile', 'mw', 'logp', 'rb', 'hba', 'hbr', 'halx', 'similarity', 'label', 'train']).T.to_csv(
            tmp_csv, index=False, header=None)
        # get smiles
        with open(src_txt, 'r') as f:
            contents = f.read().split('<EOS>\n')[:-1]  # return list
        # for each seed smile
        for seed, content in enumerate(tqdm(contents[:])):
            # seed = seed + 1297
            content = content.splitlines()
            # collate
            names, smiles, mw, logp, rb, hba, hbr, halx, similarities, labels, trains = [], [], [], [], [], [], [], [], [], [], []
            # each generated smile
            for smi_num, line in enumerate(content[:]):
                line = line.split()
                if len(line) == 11:
                    for i, lis in enumerate([smiles, mw, logp, rb, hba, hbr, halx]):
                        lis.append(line[i])
                    # float similarity
                    similarities.append(float(line[7]))
                    labels.append(float(line[8]))
                    trains.append(float(line[9]))
                    names.append(line[-1])
                else:
                    print(f'error for {line}')
            # trans to df
            df = pd.DataFrame([names, smiles, mw, logp, rb, hba, hbr, halx, similarities, labels, trains]).T
            df_seed = pd.DataFrame(df.iloc[0, :]).T  # get seed
            df = df[df.iloc[:, -3] <= 0.4]
            df.sort_values(by=8, inplace=True, ascending=True)
            df = df_seed.append(df, sort=False)
            # # add label
            # df_ac = df[df.iloc[:, 0].str.endswith('_0')]
            # df_ac['label'] = np.ones((len(df_ac)))
            # df_inac = df[~df.iloc[:, 0].str.endswith('_0')]
            # df_inac['label'] = np.zeros((len(df_inac)))
            # # merge
            # df = df_ac.append(df_inac, sort=False)
            # add to csv
            df.to_csv(tmp_csv, index=False, header=None, mode='a')
    # read csv
    print('read data from csv file')
    df = pd.read_csv(tmp_csv, encoding='utf-8')
    my_filter = properties_filer(df=df)
    # multiprocess
    pool = Pool(50)
    print('start filter....')
    result_dfs = pool.map(my_filter.name2filter, my_filter.names[:])
    pool.close()
    pool.join()
    # drop nan
    print('drop nan')
    result_dfs = list(filter(lambda x: x is not None, result_dfs))
    # merge df
    print('start merging')
    new_df = reduce(merge_2_df, result_dfs)
    new_df.columns = df.columns
    # write
    print('output to csv')
    new_df.to_csv(dst_csv, encoding='utf-8', index=False)
    print('end filtering')
