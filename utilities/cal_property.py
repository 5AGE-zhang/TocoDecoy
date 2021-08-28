#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2021/4/14 20:59
# @author : Xujun Zhang

import os
import time
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import AllChem, Descriptors, Fragments
import warnings
warnings.filterwarnings('ignore')


def cal_MW(mol):
    # cal molecule weight
    return Descriptors.MolWt(mol)


def cal_logp(mol):
    # cal molecule logp
    return Descriptors.MolLogP(mol)


def cal_HB_acceptor(mol):
    # acceptor of hydrogen bond
    return Descriptors.NumHAcceptors(mol)


def cal_HB_donor(mol):
    # donor of hydrogen bond
    return Descriptors.NumHDonors(mol)


def cal_halogen(mol):
    # count num of halogen atoms
    return Fragments.fr_halogen(mol)


def cal_rotable_bonds(mol):
    # count num of rotable bonds
    return Descriptors.NumRotatableBonds(mol)


def cal_sulfi(mol):
    # count num of S
    return Fragments.fr_sulfide(mol)


def cal_heavy_atoms(mol):
    # count num of heavy atoms
    return Descriptors.HeavyAtomCount(mol)


def cal_rings(mol):
    # count ring nums
    return Descriptors.RingCount(mol)

# init
path = r'/home/xujun/Project_4/extra_toco'  # ./
targets = os.listdir(path)
for target in targets:
    path_local = f'{path}/{target}'
    src_csv = f'{path_local}/{target}_train_lit_docked.csv'
    dst_csv = f'{path_local}/{target}_train_lit_properties.csv'
    # get smiles
    df = pd.read_csv(src_csv)
    columns = df.columns.tolist()
    df = df.values
    new_df = []
    # for each seed smile
    for idx, df_line in enumerate(tqdm(df[:])):
        try:
            mol = Chem.MolFromSmiles(df_line[0])
            tmp = np.asarray([cal_MW(mol), cal_logp(mol), cal_rotable_bonds(mol), cal_HB_acceptor(mol), cal_HB_donor(mol), cal_halogen(mol)])
            df_line = np.concatenate([df_line, tmp], axis=0)
            new_df.append(df_line)
        except:
            continue
    columns = columns + ['mw', 'logp', 'rb', 'hba', 'hbr', 'halx']
    new_df = pd.DataFrame(new_df, columns=columns)
    new_df.to_csv(dst_csv, index=False)
