#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2021/4/15 9:34
# @author : Xujun Zhang

import os
import pandas as pd
from multiprocessing import Pool

def collecte_(lig_name):
    ligand = f'{src_path}/{lig_name}'
    with open(ligand, 'r') as f:
        content = f.read()
    score = content.split('> <r_i_docking_score>')[1].split('\n')[1]
    return [lig_name.split('.')[0], score]

# collect name and docking score
path = '/home/xujun/Project_5/mix'
targets = ['active', 'test_active', 'aldh1', 'pcb', 'test_pcb', 'mapk1', 'mapk1_active', 'pcb_mapk1_train', 'pcb_mapk1_test']
for target in targets:
    src_path = f'{path}/{target}/ligands_sdf'
    dst_csv = f'{path}/{target}/docked/docking_score_from_splited_ligands.csv'
    lig_names = os.listdir(src_path)
    # multiprocess
    pool = Pool(28)
    data = pool.map(collecte_, lig_names)
    pool.close()
    pool.join()
    # pd
    df = pd.DataFrame(data, columns=['name', 'docking_score'])
    df.to_csv(dst_csv, index=False)
