#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2021/2/5 21:21
# @author : Xujun Zhang

import os
import pandas as pd
from multiprocessing import Pool
from my_utils import Smiles, docking

def trans_wrirte(mol, name):
    mol = smiler.to3D(mol)
    smiler.save_to_file(mol, file_name=name)

# init
job_type = 'aldh1'  # 'pcb' fpb
path = r'/home/Project_5/docking'
src_path = f'{path}/src'
path_undocked = f'{path}/undocked_{job_type}'
csv_file = f'{src_path}/{job_type}_filtered.csv'
# get smiles
df = pd.read_csv(csv_file, encoding='utf-8').iloc[:, :]
smiles = df.iloc[:, 1].values
names = df.iloc[:, 0].values
# trans smiles to sdf
if not os.path.exists(path_undocked):
    os.mkdir(path_undocked)
#
smiler = Smiles(smile_lis=smiles, names=names, save_path=path_undocked)
# multi process
pool = Pool(28)
pool.starmap(trans_wrirte, zip(smiler.mols, smiler.names))
pool.close()
pool.join()
# print('transform to 3D...')
# mols = pool.map(smiler.to3D, smiler.mols)
# print('write to sdf file...')
# pool.starmap(smiler.save_to_file, zip(mols, smiler.names))
# docking
protein_file = f'{src_path}/5l2n_protein.pdb'
reference_ligand = f'{src_path}/reference.mol2'
path_docked = f'{path}/docked_{job_type}'
unprepared_ligand = f'{path_undocked}/unprepared_ligand'
docked_ligand = f'{path_docked}/SP_raw'
cpu_id = 3
# mkdir
if not os.path.exists(path_docked):
    os.mkdir(path_docked)
#
docker = docking(protein_file=protein_file, reference_ligand_file=reference_ligand, src_path=path_undocked,
                 dst_path=path_docked, soft='glide')
print('merge file for preparation')
smiler.merge_file(src_files=smiler.names, dst_file=unprepared_ligand, format='sdf')
# multi process
# prepare
print('prepare ligand files...')
# pool.map(docker.prepare_ligand, smiler.names)
# if not os.path.exists(unprepared_ligand + '.mae'):
docker.prepare_ligand(file_name=unprepared_ligand.split('/')[-1], cpu_id=cpu_id)
# os.system(f'cd {path_undocked} && rm *.sdf')
# docking
print('docking ...')
# pool.map(docker.docking_vina, smiler.names)
# if not os.path.exists(docked_ligand + '.maegz'):
docker.docking_glide(unprepared_ligand, cpu_id=cpu_id)
# transform to mol2
print('transform to sdf....')
smiler.transform(src_file=docked_ligand + '.maegz', dst_file=docked_ligand + '.sdf')
# split
# print('split docked files')
# smiler.split_file(src_file=docked_ligand + '.mol2', dst_files=smiler.names, dst_path=mol2s_dir, format='mol2')
