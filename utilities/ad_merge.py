#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2021/2/19 9:38
# @author : Xujun Zhang

import argparse
import asyncio
import os
import time
from functools import partial
from multiprocessing import Pool, Process

import numpy as np
import pandas as pd
from my_utils import Smiles
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def sdf2csv(content):
    line = content[0]
    last_name = line.split('                    3D')[0].split('\n')[-2]
    score = line.split('> <r_i_docking_score>')[1].split('\n')[1]
    pd.DataFrame([last_name, score, job_type]).T.to_csv(dst_csv, index=False, header=False, mode='a')
    n = 1
    for line in content[1:]:
        lig_name = line.split('                    3D')[0].split('\n')[-2]
        if lig_name == last_name:
            lig_name = f'{lig_name}_{n}'
            n += 1
        else:
            last_name = lig_name
            n = 1
        score = line.split('> <r_i_docking_score>')[1].split('\n')[1]
        pd.DataFrame([lig_name, score, job_type]).T.to_csv(dst_csv, index=False, header=False, mode='a')

def sp2csv(src_score_file, dst_score_file):
    src_df = pd.read_csv(src_score_file)
    # get uniq_active_names
    uniq_names = list(set(i for i in src_df.loc[:, 'NAME'].values))
    # get score
    for uniq_name in uniq_names:
        tmp_df = src_df[src_df.NAME == uniq_name]
        tmp_df.sort_values(by='r_i_docking_score', inplace=True)
        tmp_df = tmp_df.loc[:, ['NAME', 'r_i_docking_score']]
        tmp_df['NAME'] = [f'{uniq_name}_{i}' for i in range(len(tmp_df))]
        tmp_df_ac = pd.DataFrame(tmp_df.iloc[0, :]).T
        if tmp_df_ac.iloc[0, 1] <= -6:
            tmp_df_decoys = tmp_df.iloc[1:, :]
            tmp_df_decoys = tmp_df_decoys[tmp_df_decoys.r_i_docking_score >= -4].iloc[-50:, :]
            tmp_df = tmp_df_ac.append(tmp_df_decoys, sort=False)
            tmp_df.to_csv(dst_score_file, index=True, header=False, mode='a')


def split_(lig_name, content, dst_path, format_):
    # lig_name = content.split('\n')[0].strip()  # 获取小分子名字
    lig_file = '{}/{}.{}'.format(dst_path, lig_name, format_)  # 定义输出分子路径
    if not os.path.exists(lig_file):
        # 输出文件
        with open(lig_file, 'w') as f:
            f.write(f'{file_label[format_]}\n' + content)

if __name__ == '__main__':
    # init
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--target', type=str, default='mpro')
    argparser.add_argument('--src_path', type=str, default='/home/xujun/Project_5/total_worflow/docking')
    argparser.add_argument('--lig_score_csv', type=str, default='score.csv')
    args = argparser.parse_args()
    # instance
    target = args.target
    src_path = f'{args.src_path}/{target}/docked'
    dst_path = f'{args.src_path}/{target}/ad'
    format_ = 'sdf'
    src_ligand_file = f'{src_path}/SP_raw.{format_}'
    src_score_file = f'{src_path}/SP_raw.csv'
    dst_csv = f'{dst_path}/{args.lig_score_csv}'
    dst_ligand = f'{dst_path}/decoy_conformations.{format_}'
    # smile
    smiler = Smiles(smile_lis=[''], names=[])
    file_label = {
        'sdf': '$$$$',
        'mol2': '@<TRIPOS>MOLECULE'
    }
    # split
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    # read file
    # 读取数据，存到con中
    with open(src_ligand_file, 'r') as f:
        con = f.read()
    # 根据@<TRIPOS>MOLECULE分割字符串  第一个是None, 根据$$$$\n分割 最后一个是‘’
    con = con.split(f'{file_label[format_]}\n')[:-1]
    # 判断数据量是否相同
    df = pd.read_csv(src_score_file)
    assert df.shape[0] == len(con), f'{target} length are not the same || df({df.shape[0]}) con({len(con)})'
    # rename
    # if not os.path.exists(dst_csv):
    pd.DataFrame(['name', 'score']).T.to_csv(dst_csv, index=False, header=False)
    # sdf2csv(con)
    sp2csv(src_score_file, dst_csv)
    # get name
    df = pd.read_csv(dst_csv, index_col=0)
    lig_names = df.name.values
    idxs = df.index.values
    con = [con[idxs[i]].replace(lig_names[i].split('_')[0], lig_names[i]) for i in range(len(idxs))]
    lig_con = '$$$$\n'.join(con)
    # output
    with open(dst_ligand, 'w') as f:
        f.write(lig_con)
