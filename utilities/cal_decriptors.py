    #!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2021/2/19 10:41
# @author : Xujun Zhang

import os
import glob
import pandas as pd
from functools import partial
from rdkit import Chem
from rdkit.Chem import AllChem
from pathos.multiprocessing import ProcessPool as Pool
from my_utils import Ifp


def cal_ecfp(mol, radius=2):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, 2048).ToBitString()


def cal_ecfp_write(lig_name, lig_smiles, similarity, label, csv_file='.'):
    tmp_mol = Chem.MolFromSmiles(lig_smiles)
    ecfp = cal_ecfp(tmp_mol)
    tmp = [lig_name] + [i for i in ecfp] + [similarity, label]
    pd.DataFrame(tmp).T.to_csv(csv_file, encoding='utf-8', index=False, header=None, mode='a')


def sp_version():
    '采用sp描述符，而不是sp分解到残基的描述符'
    # init
    job_type = 'fpb'  # fpb pcb
    path = f'/home/Project_5/cal_descriptors/v_1/{job_type}'
    src_csv = f'{path}/{job_type}_filtered.csv'
    docked_csv = f'{path}/SP.csv'
    csv_file = f'{path}/ecfp.csv'
    out_ecfp = partial(cal_ecfp_write, csv_file=csv_file)
    if not os.path.exists(csv_file):
        pd.DataFrame(['name'] + [i for i in range(2048)] + ['similarity', 'label']).T.to_csv(csv_file, encoding='utf-8', index=False, header=None)
    # get name
    df_sp = pd.read_csv(docked_csv, encoding='utf-8').dropna()
    names = df_sp.iloc[:, 0].values
    df = pd.read_csv(src_csv, encoding='utf-8')
    df.index = df.iloc[:, 0].values
    lig_smiles = df.loc[names, 'smile'].values
    similarities = df.loc[names, 'similarity'].values
    labels = df.loc[names, 'label'].values
    # add label to SP
    if 'similarity' not in df_sp.columns.tolist():
        df_sp['similarity'] = similarities
        df_sp['label'] = labels
        df_sp.to_csv(docked_csv, encoding='utf-8', index=False)
    # multiprocess
    pool = Pool(28)
    pool.map(out_ecfp, names, lig_smiles, similarities, labels)
    pool.clear()

def aldh1_version():
    '在真实aldh1数据上计算'
    # init
    job_type = 'aldh1'  # fpb pcb
    des_type = 'plec'  # ecfp
    path = f'/home/Project_5/cal_descriptors/v_1/{job_type}'
    src_csv = f'{path}/{job_type}_filtered.csv'
    docked_csv = f'{path}/SP.csv'
    csv_file = f'{path}/{des_type}.csv'
    if not os.path.exists(csv_file):
        pd.DataFrame(['name'] + [i for i in range(2048)] + ['train', 'label']).T.to_csv(csv_file, encoding='utf-8', index=False, header=None)
    # get name
    df_sp = pd.read_csv(docked_csv, encoding='utf-8').dropna()
    names = df_sp.iloc[:, 0].values
    df = pd.read_csv(src_csv, encoding='utf-8')
    df.index = df.iloc[:, 0].values
    lig_smiles = df.loc[names, 'smile'].values
    train_labels = df.loc[names, 'train'].values
    labels = df.loc[names, 'label'].values
    # add label to SP
    if 'similarity' not in df_sp.columns.tolist():
        df_sp['train'] = train_labels
        df_sp['label'] = labels
        df_sp.to_csv(docked_csv, encoding='utf-8', index=False)
    # multiprocess
    if des_type == 'ecfp':
        out_ecfp = partial(cal_ecfp_write, csv_file=csv_file)
        pool = Pool(28)
        pool.map(out_ecfp, names, lig_smiles, train_labels, labels)
        pool.clear()
    else:
        protein_file = '/home/Project_5/docking/src/5l2n_protein.pdb'
        path_mixed = f'/home/Project_5/mix/{job_type}/ligands_sdf'
        ifper = Ifp(protein_file=protein_file, lig_path=path_mixed, csv_file=csv_file, ifp_type=des_type)
        pool = Pool(28)
        pool.map(ifper.cal_ifp_2_csv, names, train_labels, labels)
        pool.clear()


def most_version():
    # init
    job_type = 'aldh1'  # fpb pcb
    des_type = 'ecfp'  # ifp  ecfp plec ifp
    path = f'/home/Project_5/cal_descriptors/v_1/{job_type}'
    src_csv = f'{path}/{job_type}_filtered.csv'
    docked_csv = f'{path}/SP_ifp.csv'
    csv_file = f'{path}/{des_type}.csv'
    print('deal SP...')
    # get name
    if job_type == 'pcb':
        idxes = [1] + list(range(15, 323))
    else:
        idxes = [1] + list(range(14, 322))
    try:
        df_sp = pd.read_csv(docked_csv, encoding='utf-8').iloc[:, idxes].dropna()
    except:
        df_sp = pd.read_csv(docked_csv, encoding='utf-8').dropna()
    names = df_sp.iloc[:, 0].values
    df = pd.read_csv(src_csv, encoding='utf-8')
    df.index = df.iloc[:, 0].values
    lig_smiles = df.loc[names, 'smile'].values
    similarities = df.loc[names, 'similarity'].values
    labels = df.loc[names, 'label'].values
    # add label to SP
    if 'similarity' not in df_sp.columns.tolist():
        df_sp['similarity'] = similarities
        df_sp['label'] = labels
        df_sp.to_csv(docked_csv, encoding='utf-8', index=False)
    print(f'cal descriptors {des_type}')
    if not os.path.exists(csv_file):
        if des_type == 'ifp':
            pd.DataFrame(['name'] + [i for i in range(3952)] + ['similarity', 'label']).T.to_csv(csv_file,
                                                                                                 encoding='utf-8',
                                                                                                 index=False,
                                                                                                 header=None)
        else:
            pd.DataFrame(['name'] + [i for i in range(2048)] + ['similarity', 'label']).T.to_csv(csv_file, encoding='utf-8', index=False, header=None)
    # multiprocess
    if des_type == 'ecfp':
        out_ecfp = partial(cal_ecfp_write, csv_file=csv_file)
        pool = Pool(28)
        pool.map(out_ecfp, names, lig_smiles, similarities, labels)
        pool.clear()
    else:
        protein_file = '/home/Project_5/docking/src/5l2n_protein.pdb'
        path_mixed = f'/home/Project_5/mix/{job_type}/ligands'
        ifper = Ifp(protein_file=protein_file, lig_path=path_mixed, csv_file=csv_file, ifp_type=des_type)
        pool = Pool(28)
        pool.map(ifper.cal_ifp_2_csv, names, similarities, labels)
        pool.clear()

if __name__ == '__main__':
    aldh1_version()
