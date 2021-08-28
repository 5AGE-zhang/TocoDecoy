#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2021/1/26 21:19
# @author : Xujun Zhang

import os
import sys
import time
import warnings
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crnn'))
import argparse
import numpy as np
from my_utils import mol_controller, append2csv
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import AllChem
from ddc_pub import ddc_v3 as ddc
# close warning
warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
# set GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Jobcontroller():
    def __init__(self, data_file, result_file):
        self.model_type = 'pcb'  # 'pcb'fpb
        self.file_path = os.path.dirname(os.path.realpath(__file__))  
        self.model_file = f'{self.file_path}/models/{self.model_type}_model'
        # self.model = ddc.DDC(model_name=self.model_file)
        self.data_file = data_file
        self.result_file = result_file  # f'{self.data_path}/result/1653_1894_test.txt'
        self.end_token = '<EOS>\n'

    def read_source_file(self):
        smiles, names = [], []
        # read
        with open(self.data_file, 'r') as f:
            content = f.read().splitlines()
        # collate
        for line in content:
            line = line.split()
            smiles.append(line[0])
            names.append(line[1])
        return smiles, names

class Sampler(Jobcontroller):
    def __init__(self, data_file, result_file, seed_smile, seed_name, seed_label):
        super(Sampler, self).__init__(data_file, result_file)
        self.seed_smile = seed_smile
        self.seed_name = seed_name
        self.seed_mol = Chem.MolFromSmiles(self.seed_smile)
        self.seed_label = seed_label
        self.generated_mols = []

    def sample_batch(self, model, tmp, model_type='fpb'):
        '''

        :return: list of generated ligand in format of rdkit.mol
        '''
        # collate seed smile
        if model_type == 'fpb':
            # cal ecfp as latent Z
            latant_z = np.array([AllChem.GetMorganFingerprintAsBitVect(self.seed_mol, 2, 2048)]).repeat(1024, axis=0)
        elif model_type == 'pcb':
            # cal properties as latent Z
            moler = mol_controller(mols=[self.seed_mol])
            latant_z = moler.cal_properties().repeat(1024, axis=0)
        # generate smiles
        smiles, _ = model.predict_batch(latent=latant_z, temp=tmp)  # self.model.predict_batch(latent=ecfps, temp=temp)
        print(f'generate smiles: {len(smiles)}')
        unique_smiles = set(smiles)
        # To compare the results, convert smiles_out to CANONICAL
        self.generated_mols.extend([Chem.MolFromSmiles(smile) for smile in unique_smiles])
        # return self.generated_mols

    def self_check(self):
        moler = mol_controller(mols= [self.seed_mol])
        if moler.unique_num == 0:
            return False
        else:
            return True

    def check_uniq(self):
        # calculate similarity
        moler = mol_controller(mols=self.generated_mols)
        return moler.unique_num

    def cal_similaritiesANDproperties(self):
        '''

        :param generated_mols: ist of generated ligand in format of rdkit.mol [1, 2, ]
        :return: [  ....
                    [ MW, logp, rotable_bonds, HBacceptor, HBdonor, halogen, similarities]
                    ....
                    ] (256, 7)
        '''
        # conclude seed smile for property calculation
        self.generated_mols.insert(0, self.seed_mol)
        moler = mol_controller(mols=self.generated_mols)
        # cal similarity
        similarities = moler.cal_similarity(seed_mol=self.seed_mol)
        # cal properties
        self.properties_similarities = moler.cal_properties(labels=similarities)
        # all smiles = seed smile + generated smiles
        self.all_smiles = moler.smiles
        # return properties_similarities

    def write2txt(self):
        # init
        content = f'''{self.seed_smile}'''
        # first seed smile
        for i in self.properties_similarities[0]:
            content += f' {i:.2f}'
        content += f' {self.seed_label} {self.seed_name}_0\n'
        # generated
        for i in range(1, len(self.properties_similarities)):
            # smile
            content += f'{self.all_smiles[i]}'
            property_similarity = self.properties_similarities[i]
            # add property
            for j in property_similarity:
                content += f' {j:.2f}'
            content += f' {0} {self.seed_name}_{i}\n'
        # add end
        content += self.end_token
        # write
        with open(self.result_file, 'a') as f:
            f.write(content)



if __name__ == '__main__':
    # init
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--src_file', type=str, default='/path/molecular_generation/data/source/dude.txt')
    argparser.add_argument('--dst_file', type=str, default='/path/molecular_generation/data/result/du_decoys.txt')
    argparser.add_argument('--target_num_for_each_smi', type=int, default=200)
    args = argparser.parse_args()
    #
    data_file = f'{args.src_file}'
    result_file = f'{args.dst_file}'
    this_job = Jobcontroller(data_file, result_file)
    target_num_for_each_smi = args.target_num_for_each_smi
    # remove exists result file
    if os.path.exists(this_job.result_file):
        os.remove(this_job.result_file)
    # read source file
    seed_smiles, seed_names = this_job.read_source_file()
    sample_model = ddc.DDC(model_name=this_job.model_file)
    # generate
    start_time = time.time()
    for i, seed_smile in enumerate(seed_smiles[:]):
        each_time = time.time()
        print(f'\n-------sample for {seed_smile} idx: {i} / {len(seed_names)}-------')
        # init
        seed_label = 1
        sampler = Sampler(data_file, result_file, seed_smile, seed_names[i], seed_label)
        if sampler.self_check():
            unique_num = n = 0
            # generating 200 molecules for each seed smiles
            while unique_num < target_num_for_each_smi:
                print(f'----------repeat: {n} ----------')
                sampler.sample_batch(model=sample_model, tmp=1, model_type=this_job.model_type)
                unique_num = sampler.check_uniq()
                n += 1
                if n >= 29:
                    break
            # cal propertires and similarities
            properties_similarities = sampler.cal_similaritiesANDproperties()
            # write to file
            sampler.write2txt()
            print(f'Time Cost: {(time.time()-each_time)/60:.3f} min')
            print(f'Total Time: {(time.time()-start_time)/60:.3f} min')
            print('-------------------End-------------------')
        else:
            print('bad smile')


