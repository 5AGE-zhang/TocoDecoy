#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2021/8/3 9:18
# @author : Xujun Zhang

import argparse
import os
import pickle
from functools import partial
from multiprocessing import Pool
import torch
import prody
from graph_constructor import graphs_from_mol_mul
from rdkit import Chem
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (1000000, rlimit[1]))


def generate_complex(lig_name, ligand_obj, lig_label, active_path, decoys_path):
    try:
        dst_path = active_path if lig_label == 1 else decoys_path
        dst_file = f'{dst_path}/{lig_name}.pdb'
        if not os.path.exists(dst_file):
            # select residues that have at least an atom within 10 A of ligand molecule.
            selected = protein_obj.select('same residue as within 10 of ligand',
                                          ligand=ligand_obj.GetConformer().GetPositions())
            prody.writePDB(dst_file, selected)
        pocket = Chem.MolFromPDBFile(dst_file)
        return ligand_obj, pocket, lig_name, lig_label
    except:
        print(lig_name)
        return None


def generate_decoy_complex(lig_name, ligand_obj, dst_path):
    try:
        lig_label = 0
        dst_file = f'{dst_path}/{lig_name}.pdb'
        if not os.path.exists(dst_file):
            # select residues that have at least an atom within 10 A of ligand molecule.
            selected = protein_obj.select('same residue as within 10 of ligand',
                                          ligand=ligand_obj.GetConformer().GetPositions())
            prody.writePDB(dst_file, selected)
        pocket = Chem.MolFromPDBFile(dst_file)
        return ligand_obj, pocket, lig_name, lig_label
    except:
        return None


# init
argparser = argparse.ArgumentParser()
argparser.add_argument('--target', type=str)
argparser.add_argument('--job_type', type=str)  # job_types = ['train_cd', 'test_cd', 'train_top50', 'train_300', 'train_1000', 'train_lit', 'test_cd', 'test_filtered', 'test_lit']
argparser.add_argument('--lig_path', type=str, default='path to ligand')
argparser.add_argument('--pocket_path', type=str, default='path for save generated pocket')
argparser.add_argument('--graph_path', type=str, default='path for save generated graph')
args = argparser.parse_args()
# parse
lig_path = args.lig_path
pocket_path = args.pocket_path
target = args.target
job_type = args.job_type
lig_path_local = f'{lig_path}/{target}'
pocket_path_local = f'{pocket_path}/{target}'
graph_path = args.graph_path
graph_path_local = f'{graph_path}/{target}'
os.makedirs(graph_path_local, exist_ok=True)
os.makedirs(pocket_path_local, exist_ok=True)
protein_file = f'{lig_path_local}/{target}_protein.pdb'
protein_obj = prody.parsePDB(protein_file)
de_graph = f'{graph_path_local}/{job_type}.pkl'
if __name__ == '__main__':
    if not os.path.exists(de_graph):
        pool = Pool()
        # cd
        if 'cd' in job_type:
            ligand_file = f'{lig_path_local}/{target}_{job_type}_sp_raw.sdf'
            mols = Chem.SDMolSupplier(ligand_file)
            mols = list(filter(lambda x: x is not None, mols))
            lig_names = [mol.GetProp('_Name') for mol in mols]
            labels = [1 if lig_name.endswith('_0') else 0 for lig_name in lig_names]
            pocket_path_local_local1 = f'{pocket_path_local}/active'
            pocket_path_local_local2 = f'{pocket_path_local}/cd'
            for dir_ in [pocket_path_local_local1, pocket_path_local_local2]:
                os.makedirs(dir_, exist_ok=True)
            generate_fn = partial(generate_complex, active_path=pocket_path_local_local1, decoys_path=pocket_path_local_local2)
            data_in = zip(lig_names, mols, labels)
            # generate pocket
            cd_data = pool.starmap(generate_fn, data_in)
            cd_data = list(filter(lambda x: x is not None, cd_data))
        else:
            # td
            ligand_file = f'{lig_path_local}/{target}_{job_type}_sp_raw.sdf'
            mols = Chem.SDMolSupplier(ligand_file)
            mols = list(filter(lambda x: x is not None, mols))
            lig_names = [mol.GetProp('_Name') for mol in mols]
            # labels = [1 if lig_name.endswith('_0') else 0 for lig_name in lig_names]
            pocket_path_local_local = f'{pocket_path_local}/{job_type}'
            os.makedirs(pocket_path_local_local, exist_ok=True)
            generate_fn = partial(generate_decoy_complex, dst_path=pocket_path_local_local)
            data_in = zip(lig_names, mols)
            # generate pocket
            cd_data = pool.starmap(generate_fn, data_in)
        # generate graph
        graphs_from_mol_mul_ = partial(graphs_from_mol_mul, dst_path=graph_path_local, job_type=job_type)
        graphs = pool.starmap(graphs_from_mol_mul_, cd_data)
        pool.close()
        pool.join()
