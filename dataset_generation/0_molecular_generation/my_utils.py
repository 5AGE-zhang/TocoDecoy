#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2021/1/27 17:21
# @author : Xujun Zhang

import csv
import os

import numpy as np
import h5py
from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger
from rdkit.Chem import AllChem, Descriptors, Fragments

RDLogger.DisableLog('rdApp.*')


class mol_controller():
    def __init__(self, smiles=None, mols=None):
        '''

        :param smiles: list of smile
        :param mols: list of rdkit mol
        '''
        # init
        if mols and smiles:
            self.mols = mols
            self.smiles = smiles
            self.checkFrommols()
        elif mols:
            self.mols = mols
            self.checkFrommols()
        elif smiles:
            self.smiles = smiles
            self.checkFromsmiles()
        else:
            print('initializing filed for None smiles and mols')
            raise SyntaxError

    def checkFrommols(self):
        smiles = []
        self.unparsable_smiles_idx = []
        for idx, mol in enumerate(self.mols):
            # 从 smiles 转成 mol 没有报错，但是从 mol 转成 smiles 可能会报错 rdkit.Chem.rdmolfiles.MolToSmiles(NoneType)
            try:
                smiles.append(Chem.MolToSmiles(mol))
            except:
                self.unparsable_smiles_idx.append(idx)
        print(f'unparsable: {len(self.unparsable_smiles_idx)}')
        print(f'valid smiles: {len(smiles)}')
        # remove duplicate smiles
        self.smiles = list(set(smiles))  #
        self.smiles.sort(key=smiles.index)  # sort to make rank united
        self.mols = [Chem.MolFromSmiles(smile) for smile in self.smiles]
        self.unique_num = len(self.smiles)
        print(f'unique smiles: {len(self.smiles)}')

    def checkFromsmiles(self):
        self.mols = []
        self.unparsable_smiles_idx = []
        for idx, smile in enumerate(self.smiles):
            try:
                self.mols.append(Chem.MolFromSmiles(smile))
            except:
                self.unparsable_smiles_idx.append(idx)
        print(f'unparsable: {len(self.unparsable_smiles_idx)}')
        self.checkFrommols()

    def cal_similarity(self, seed_mol):
        # init
        smis = []
        # cal ecfp
        radius = 2
        seed_fp = AllChem.GetMorganFingerprint(seed_mol, radius)
        # cal smis
        print('start cal similarity....')
        # for mol in tqdm(self.mols):
        self.unparsable_smiles_idx = []
        for i, mol in enumerate(self.mols):
            try:
                # cal fp
                fp = AllChem.GetMorganFingerprint(mol, radius)
                # cal smi
                smi = DataStructs.DiceSimilarity(seed_fp, fp)
                # apped smi
                smis.append(smi)
            except:
                self.unparsable_smiles_idx.append(i)
        # remobe bad smiles
        for i in reversed(self.unparsable_smiles_idx):
            self.mols.pop(i)
            self.smiles.pop(i)
        # to numpy
        smis = np.array(smis)
        # print
        print(f'similarity: mean {smis.mean():.3f} || min {smis.min():.3f} || max {smis.max():.3f}')
        return smis

    def cal_MW(self):
        # cal molecule weight
        return [Descriptors.MolWt(mol) for mol in self.mols]

    def cal_logp(self):
        # cal molecule logp
        return [Descriptors.MolLogP(mol) for mol in self.mols]

    def cal_HB_acceptor(self):
        # acceptor of hydrogen bond
        return [Descriptors.NumHAcceptors(mol) for mol in self.mols]

    def cal_HB_donor(self):
        # donor of hydrogen bond
        return [Descriptors.NumHDonors(mol) for mol in self.mols]

    def cal_halogen(self):
        # count num of halogen atoms
        return [Fragments.fr_halogen(mol) for mol in self.mols]

    def cal_rotable_bonds(self):
        # count num of rotable bonds
        return [Descriptors.NumRotatableBonds(mol) for mol in self.mols]

    def cal_sulfi(self):
        # count num of S
        return [Fragments.fr_sulfide(mol) for mol in self.mols]

    def cal_heavy_atoms(self):
        # count num of heavy atoms
        return [Descriptors.HeavyAtomCount(mol) for mol in self.mols]

    def cal_rings(self):
        # count ring nums
        return [Descriptors.RingCount(mol) for mol in self.mols]

    def cal_properties(self, labels=None):
        '''
        cal properties for mols (MW, logp, rotable_bonds, HBacceptor, HBdonor, halogen)
        if labels are given, merge labels and mols
        :param labels: (batch_size, 1)
        :return: (batch_size, 7)
        '''
        try:
            if labels is not None:
                return np.array(list(zip(self.cal_MW(), self.cal_logp(), self.cal_rotable_bonds(), self.cal_HB_acceptor(),
                                         self.cal_HB_donor(), self.cal_halogen(), labels)))
                # return np.concatenate(
                #     [np.array(list(zip(self.cal_MW(), self.cal_logp(), self.cal_rotable_bonds(), self.cal_HB_acceptor(),
                #                        self.cal_HB_donor(), self.cal_halogen()))), labels], axis=-1)
            else:
                return np.array(list(zip(self.cal_MW(), self.cal_logp(), self.cal_rotable_bonds(), self.cal_HB_acceptor(),
                                         self.cal_HB_donor(), self.cal_halogen())))
        except:
            return np.ones((len(self.mols), 7))


class Smiles():
    def __init__(self, smile_lis, names=None, save_path='./'):
        self.smiles = smile_lis
        self.mols = [Chem.MolFromSmiles(smile) for smile in self.smiles]
        self.names = names if names else list(range(len(self.smiles)))
        self.path = save_path
        self.file_label = {
            'sdf': '$$$$',
            'mol2': '@<TRIPOS>MOLECULE'
        }

    def to3D(self, mol):
        # add H
        mol = Chem.AddHs(mol)
        # to 3D
        AllChem.EmbedMolecule(mol)
        # delete H
        # mol = Chem.RemoveHs(mol)
        return mol

    def save_to_file(self, mol, file_name, format='sdf'):
        # file
        sdf_file = f'{self.path}/{file_name}.sdf'
        # write file to sdf
        Chem.MolToMolFile(mol, sdf_file)
        if format == 'mol2':
            mol2_file = f'{self.path}/{file_name}.mol2'
            # trans2 mol2
            cmd = 'module load openbabel && obabel {0} -O {1}'.format(sdf_file, mol2_file)
            os.system(cmd)

    def merge_file(self, src_files, dst_file, format='sdf'):
        # init
        content = ''
        # for-loop
        for src_file in src_files:
            src_file_full = f'{self.path}/{src_file}.{format}'
            with open(src_file_full, 'r') as f:
                content += f'{src_file}{f.read()}\n$$$$\n'
        # output file
        dst_file = f'{dst_file}.{format}'
        with open(dst_file, 'w') as f:
            f.write(content)

    def split_file(self, src_file, dst_files, dst_path, format='sdf'):
        '''

        :param src_file: src_path/file.sdf
        :param dst_files: [file_name, file_name1...]
        :param dst_path:
        :param format:
        :return:
        '''
        # 读取数据，存到con中
        with open(src_file, 'r') as f:
            con = f.read()
        # 根据@<TRIPOS>MOLECULE分割字符串
        con = con.split(f'{self.file_label[format]}\n')
        for i in range(0, len(con)):
            if con[i] != '':
                lig_name = con[i].split('\n')[0].strip()  # 获取小分子名字
                lig_file = '{}/{}.{}'.format(dst_path, lig_name, format)  # 定义输出分子路径

                # 递归检查重名在文件
                def same_file(lig_file, n=0):
                    if os.path.exists(lig_file):
                        n += 1
                        lig_file = '{}/{}_{}.{}'.format(dst_path, lig_name, n, format)
                        return same_file(lig_file, n)
                    else:
                        return lig_file

                lig_file = same_file(lig_file)  # 检查是否重名
                # 输出文件
                with open(lig_file, 'w') as f:
                    if format == 'sdf':
                        f.write(con[i] + f'{self.file_label[format]}\n')
                    else:
                        # format == 'mol2'
                        f.write(f'{self.file_label[format]}\n' + con[i])

    def transform(self, src_file, dst_file):
        cmd = f'module load schrodinger && structconvert {src_file} {dst_file}'
        os.system(cmd)


class Pharmacophore():
    def __init__(self, soft='openeye'):
        if soft == 'openeye':
            self.load_module = 'module load openeye/applications-2018.11.3 &&'
            self.phase_screen = 'rocs -dbase {0} -query {1} -cutoff 1.0 -nostructs -prefix {2}'  # dbase unscreened file  # result file = prefix_1.rpt  prefix_hits_1.sdf
        elif soft =='schrodinger':
            self.load_module = 'module load schrodinger/2017-4 &&'
            self.generate_hypo = 'create_hypoFiles {0} {1}'  # 0 infile {reference file} 1 hypoID{output file name}
            self.generate_hypo_mul = 'phase_hypothesis {0} -HOST "localhost:28" '  # phase_hypothesis phase_pharm_8.inp -HOST "localhost:16"
            self.phase_screen = 'phase_screen {0} {1} {2} -distinct -nosort -report 1 -WAIT -HOST "localhost:28" -TMPLAUNCHDIR -ATTACHED'  # 0 unscreened file 1 hypo_file 2 job name  Hits will be returned in <jobName>-hits.maegz. -distinct -nosort
            self.phase_screen_min = 'phase_screen {0} {1} {2} -refine -force_field OPLS3 -nosort -report 1 -match 6 -WAIT -HOST "localhost:28" -TMPLAUNCHDIR -ATTACHED'  # Generate conformers on-the-fly for the highest scoring match and search for additional matches. Not  subject to the above restrictions on -flex, but not  valid in combination with -flex.
            self.default_def = './phase_pharm.def'
            self.phase_pharm_inp = '''INPUT_STRUCTURE_FILE   {0}  # phase_pharm_8.maegz
    USE_PREALIGNED_LIGANDS   False
    USE_FAST_SCORING   False
    USE_LIGAND_GROUPING   True
    LIGAND_PERCEPTION   stereo
    GENERATE_CONFORMERS   True
    MAX_NUMBER_CONFORMERS   50
    USE_CONFORMER_MINIMIZATION   True
    REQUIRED_MATCH_FRACTION   {1}  # 0.5
    NUMBER_FEATURES_PER_HYPOTHESIS   {2},{3}  # 4, 5
    PREFERRED_MIN_SITES   {4}  # 5
    HYPOTHESIS_DIFFERENCE_CRITERIA   0.5
    HYPOTHESES_KEPT_PER_FEATURE_SIZE   10
    FEATURE_FREQUENCY_A   0, 3
    FEATURE_FREQUENCY_D   0, 3
    FEATURE_FREQUENCY_H   0, 3
    FEATURE_FREQUENCY_R   0, 3
    FEATURE_FREQUENCY_P   0, 3
    FEATURE_FREQUENCY_N   0, 3
    FEATURE_FREQUENCY_X   0, 3
    FEATURE_FREQUENCY_Y   0, 3
    FEATURE_FREQUENCY_Z   0, 3
    FEATURE_DEFINITION_FILE   {5}  # phase_pharm.def
    SCORE_WEIGHT_VECTOR   1.0
    SCORE_WEIGHT_SITE   1.0
    SCORE_WEIGHT_VOLUME   1.0
    SCORE_WEIGHT_SELECTIVITY   1.0
    SCORE_WEIGHT_LOG_MATCH   1.0
    SCORE_WEIGHT_INACTIVE   1.0
    SCORE_WEIGHT_SURVIVAL   0.06
    SCORE_WEIGHT_BEDROC   1.0
    FEATURE_TOLERANCE_A   2.0
    FEATURE_TOLERANCE_D   2.0
    FEATURE_TOLERANCE_H   2.0
    FEATURE_TOLERANCE_R   2.0
    FEATURE_TOLERANCE_P   2.0
    FEATURE_TOLERANCE_N   2.0
    FEATURE_TOLERANCE_X   2.0
    FEATURE_TOLERANCE_Y   2.0
    FEATURE_TOLERANCE_Z   2.0
    APPEND_EXCLUDED_VOLUMES   False
    '''

    def generate_hypo_file(self, path, seed_file, hypo_prefix):
        cmd = f'cd {path} && {self.load_module + self.generate_hypo.format(seed_file, hypo_prefix)}'
        # print(cmd)
        os.system(cmd)

    def phase_screen_file(self, path, unscreened_file, hypo_file, hypo_prefix):
        cmd = f'cd {path} && {self.load_module + self.phase_screen.format(unscreened_file, hypo_file, hypo_prefix)}'
        # print(cmd)
        os.system(cmd)



def append2csv(csv_file, new_lis):
    '''

    :param csv_file: csv file
    :param new_lis: list waited to be added to the end of the csv file
    :return:
    '''

    csv_ = open(csv_file, 'a')
    csv_writer = csv.writer(csv_)
    csv_writer.writerow(new_lis)
    csv_.close()


def append2txt(txt_file, src_file):
    with open(src_file, 'r') as f:
        content = f.read()
    with open(txt_file, 'a') as f:
        f.write(content)

def load_dataset(data_file):
    dataset = h5py.File(data_file, "r")
    binary_mols = dataset["mols"][:]
    dataset.close()
    mols = [Chem.Mol(binary_mol) for binary_mol in binary_mols]
    return mols