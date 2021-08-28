#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2021/1/27 17:21
# @author : Xujun Zhang

import csv
import os
import pandas as pd
# import h5py
import numpy as np
import oddt
from oddt import fingerprints
from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger
from rdkit.Chem import AllChem, Descriptors, Fragments
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
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
        elif mols is not None:
            self.mols = mols
            self.checkFrommols()
        elif smiles is not None:
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


    def cal_ecfp(self, mol, radius=2):
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius)

    def cal_ecfp_write(self, mol2_file, csv_file):
        name = os.path.basename(mol2_file).split('.')[0]
        tmp_mol = Chem.MolFromMol2File(mol2_file)
        ecfp = self.cal_ecfp(tmp_mol).tolist()
        tmp = [name] + ecfp
        pd.DataFrame(tmp).T.to_csv(csv_file, encoding='utf-8', index=False, header=None, mode='a')

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
        if labels is not None:
            return np.array(list(zip(self.cal_MW(), self.cal_logp(), self.cal_rotable_bonds(), self.cal_HB_acceptor(),
                                     self.cal_HB_donor(), self.cal_halogen(), labels)))
            # return np.concatenate(
            #     [np.array(list(zip(self.cal_MW(), self.cal_logp(), self.cal_rotable_bonds(), self.cal_HB_acceptor(),
            #                        self.cal_HB_donor(), self.cal_halogen()))), labels], axis=-1)
        else:
            return np.array(list(zip(self.cal_MW(), self.cal_logp(), self.cal_rotable_bonds(), self.cal_HB_acceptor(),
                                     self.cal_HB_donor(), self.cal_halogen())))


class Smiles():
    def __init__(self, smile_lis, names=None, save_path='./'):
        self.smiles = smile_lis
        self.mols = [Chem.MolFromSmiles(smile) for smile in self.smiles]
        self.names = names if len(names) != 0 else list(range(len(self.smiles)))
        self.path = save_path
        self.file_label = {
            'sdf': '$$$$',
            'mol2': '@<TRIPOS>MOLECULE'
        }

    def to3D(self, mol):
        try:
            # add H
            mol = Chem.AddHs(mol)
            # to 3D
            AllChem.EmbedMolecule(mol)
            # delete H
            # mol = Chem.RemoveHs(mol)
        except:
            mol = 'bad'
        return mol

    def save_to_file(self, mol, file_name, format='sdf'):
        if mol != 'bad':
            # file
            sdf_file = f'{self.path}/{file_name}.sdf'
            # write file to sdf
            Chem.MolToMolFile(mol, sdf_file)
            if format == 'mol2':
                mol2_file = f'{self.path}/{file_name}.mol2'
                # trans2 mol2
                cmd = 'module load openbabel && obabel {0} -O {1}'.format(sdf_file, mol2_file)
                os.system(cmd)
            return file_name
        else:
            return None

    def merge_file(self, src_files, dst_file, format='sdf'):
        # init
        content = ''
        # for-loop
        for src_file in src_files:
            src_file_full = f'{self.path}/{src_file}.{format}'
            try:
                with open(src_file_full, 'r') as f:
                    content += f'{src_file}{f.read()}\n$$$$\n'
            except:
                print(f'{src_file_full} not exist')
        # output file
        dst_file = f'{dst_file}.{format}'
        with open(dst_file, 'w') as f:
            f.write(content)

    def split_(self, content, dst_path, format):
        if content != '':
            lig_name = content.split('\n')[0].strip()  # 获取小分子名字
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
                    f.write(content + f'{self.file_label[format]}\n')
                else:
                    # format == 'mol2'
                    f.write(f'{self.file_label[format]}\n' + content)

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
        for i in tqdm(range(0, len(con))):
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


    def split_file_paral(self, src_file, dst_path, format='sdf'):
        # 读取数据，存到con中
        with open(src_file, 'r') as f:
            con = f.read()
        # 根据@<TRIPOS>MOLECULE分割字符串
        con = con.split(f'{self.file_label[format]}\n')
        # multiprocessing
        pool = Pool(28)
        split_ = partial(self.split_, dst_path=dst_path, format=format)
        pool.map(split_, con)
        pool.close()
        pool.join()


    def transform(self, src_file, dst_file):
        cmd = f'module load schrodinger && structconvert {src_file} {dst_file}'
        os.system(cmd)


class Pharmacophore():
    def __init__(self, soft='openeye'):
        if soft == 'openeye':
            self.load_module = 'module load openeye/applications-2018.11.3 &&'
            self.phase_screen = 'rocs -dbase {0} -query {1} -cutoff 1.0 -nostructs -prefix {2}'  # dbase unscreened file  # result file = prefix_1.rpt  prefix_hits_1.sdf
        elif soft == 'schrodinger':
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
        try:
            os.system(cmd)
        except:
            print('screen error')


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


def get_label(report_file, valida_names, filter_color=True):
    if os.path.exists(report_file):
        # read
        with open(report_file, 'r') as f:
            content = f.read().splitlines()[1:]
        # init labels to zeros
        labels = np.zeros((len(valida_names)))
        if filter_color:
            for i, line in enumerate(content):
                line = line.split()
                name = line[0]
                color = float(line[5])
                if color > 0.5:
                    try:
                        idx = valida_names.index(name)
                    except:
                        name = name.split('_')
                        seed = name[0]
                        num = name[1]
                        name = f'{seed}_{num}'
                        idx = valida_names.index(name)
                    labels[idx] = 1
        else:
            # get names
            for i, line in enumerate(content):
                line = line.split()
                name = line[0]
                try:
                    idx = valida_names.index(name)
                except:
                    name = name.split('_')
                    seed = name[0]
                    num = name[1]
                    name = f'{seed}_{num}'
                    idx = valida_names.index(name)
                labels[idx] = 1
        return labels


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


class properties_filer():
    def __init__(self, df, mw=40, logp=1.5, rb=1, hba=1, hbd=1, halx=1):
        self.df = df
        self.names = self.get_uniq_name()
        self.properties_range = [mw, logp, rb, hba, hbd, halx]

    def get_uniq_name(self):
        # get names
        names = self.df.iloc[:, 0].values
        names = (i.split('_')[0] for i in names)
        names = list(set(names))
        return names

    def Pfilter(self, df_tmp, padding=2, label=0):
        '''

        :param df_tmp: name  smile 	mw 	logp 	rb 	hba 	hbr 	halx 	similatity 	label
                        0_0
                        0_1
                        0_2
                        ...
                        0_n
        :param padding: df.iloc[:, 2] = df.loc[:, 'mw']  >>> padding = 2
        :return:
        '''
        if len(df_tmp):
            df_seed = pd.DataFrame(df_tmp.iloc[0, :]).T
            if len(df_tmp):
                # properties
                for j, k in enumerate(self.properties_range):
                    df_tmp = df_tmp[(df_tmp.iloc[:, j + padding] >= (float(df_tmp.iloc[0, j + padding]) - k)) & (
                                df_tmp.iloc[:, j + padding] <= (float(df_tmp.iloc[0, j + padding]) + k))]
                # label
                df_tmp = df_tmp[df_tmp.label.values == label]
                # append
                df_tmp = df_seed.append(df_tmp, sort=False)
                return df_tmp

    def name2filter(self, molecule_name, padding=2, label=0):
        df = self.df[self.df.iloc[:, 0].str.startswith(f'{molecule_name}_')]
        return self.Pfilter(df_tmp=df, padding=padding, label=label)

    def get_top_n(self, molecule_name, top_n, ascending=True):
        df = self.df[self.df.iloc[:, 0].str.startswith(f'{molecule_name}_')]
        df_seed = pd.DataFrame(df.iloc[0, :]).T
        df_decoys = df.iloc[1:, :]
        df_decoys.sort_values(by='similarity', inplace=True, ascending=ascending)
        df_decoys = df_decoys.iloc[:top_n, :]
        df = df_decoys.append(df_seed, sort=False)
        return df


def merge_2_df(df1, df2):
    new_df = df1.append(df2, sort=False)
    return new_df

class docking():
    def __init__(self, protein_file, reference_ligand_file, src_path='./', dst_path='./', soft='vina'):
        self.src_path = src_path
        self.dst_path = dst_path
        self.protein_file = protein_file
        self.protein_path = os.path.dirname(protein_file)
        self.reference_ligand = reference_ligand_file
        self.protein_no_suffix = os.path.splitext(protein_file)[0]
        # get pocket
        self.x, self.y, self.z = self.get_xyz()
        # cmd
        if soft == 'vina':
            self.protein_suffix = 'pdbqt'
            self.ligand_prepare_cmd = 'cd {0} && source deactivate && module purge && module load vina && prepare_ligand4.py -l {0}/{1}.mol2'
            self.receptor_prepare_cmd = 'cd {0} && source deactivate && module purge && module load vina && prepare_receptor4.py -r {1} -o {2}'
            self.pred_protein = f'{self.protein_path}/{os.path.basename(protein_file).split(".")[0]}.{self.protein_suffix}'
            # prepare
            if not os.path.exists(self.pred_protein):
                self.prepare_protein(protein_file)
        elif soft == 'glide':
            self.protein_suffix = 'mae'
            self.ligand_prepare_cmd = 'cd {0} && module load schrodinger && ligprep -i 0 -nt -s 1 -g -isd {0}/{1}.sdf -WAIT -HOST "cu0{2}:28" -omae {0}/{1}.mae'
            self.receptor_prepare_cmd = 'cd {0} && module load schrodinger && prepwizard -disulfides -fillsidechains -fillloops -epik_pH 7.0 -epik_pHt 0 -watdist 0 -samplewater -propka_pH 7.0 -delwater_hbond_cutoff 3 -rmsd 0.3 -fix -f 3 -WAIT -NOJOBID {1} {2}'
            self.pred_protein = f'{self.protein_path}/{os.path.basename(protein_file).split(".")[0]}.{self.protein_suffix}'
            self.grid_in = f'{self.protein_path}/grid.in'
            self.grid_zip = f'{self.protein_path}/glide-grid.zip'
            # prepare
            if not os.path.exists(self.pred_protein):
                self.prepare_protein(protein_file)
            # grid_generation
            if not os.path.exists(self.grid_zip):
                self.generate_grid()
            # docking set
            self.glide_in = f'{dst_path}/SP.in'

    def prepare_ligand(self, file_name, cpu_id):
        # cmd conda deactivate &&
        cmd = self.ligand_prepare_cmd.format(self.src_path, file_name, cpu_id)
        os.system(cmd)

    def prepare_protein(self, protein_file=None):
        if protein_file is None:
            protein_file = self.protein_file
        # cmd
        cmd = self.receptor_prepare_cmd.format(self.protein_path, protein_file, self.pred_protein)
        os.system(cmd)

    def get_xyz(self):
        # 根据共晶配体确定坐标 mol2
        x = os.popen(
            "cat %s | sed -n '/@<TRIPOS>ATOM/,/@<TRIPOS>BOND/'p | awk '{print $3}' | awk '{x+=$1} END {print x/(NR-2)}'" % self.reference_ligand).read()
        y = os.popen(
            "cat %s | sed -n '/@<TRIPOS>ATOM/,/@<TRIPOS>BOND/'p | awk '{print $4}' | awk '{y+=$1} END {print y/(NR-2)}'" % self.reference_ligand).read()
        z = os.popen(
            "cat %s | sed -n '/@<TRIPOS>ATOM/,/@<TRIPOS>BOND/'p | awk '{print $5}' | awk '{z+=$1} END {print z/(NR-2)}'" % self.reference_ligand).read()

        return float(x.strip()), float(y.strip()), float(z.strip())

    def generate_grid(self):
        outline = '''GRID_CENTER   {0}, {1}, {2}
GRIDFILE   glide-grid.zip
INNERBOX   10, 10, 10
OUTERBOX   30, 30, 30
RECEP_FILE   {3}
RECEP_VSCALE   1.0
RECEP_CCUT   0.25
            '''.format(self.x, self.y, self.z, self.pred_protein)
        with open(self.grid_in, 'w') as f:
            f.write(outline)
        # 生成格点
        cmd = 'cd {} && module load schrodinger && glide grid.in -NOJOBID -WAIT'.format(self.protein_path)
        os.system(cmd)

    def docking_vina(self, ligand_name):
        # cmd
        cmd = f'/home/xujun/Soft/Score_Function/smina/smina --receptor {self.pred_protein} --ligand {self.src_path}/{ligand_name}.pdbqt -o {self.dst_path}/{ligand_name}.mol2 --center_x {self.x} --center_y {self.y} --center_z {self.z} --size_x 18.75 --size_y 18.75 --size_z 18.75 --num_modes 1 --cpu 1 -q'
        os.system(cmd)

    def docking_glide(self, ligand_file, cpu_id):
        # write in
        content = '''GRIDFILE   {0}
LIGANDFILE   {1}.mae
POSES_PER_LIG   1
POSE_OUTTYPE   ligandlib
NOSORT TRUE
DOCKING_METHOD confgen
PRECISION   SP
WRITE_CSV True
WRITE_RES_INTERACTION True'''.format(self.grid_zip, ligand_file)
        with open(self.glide_in, 'w') as f:
            f.write(content)
        # exit
        cmd = 'cd {0} && module load schrodinger && glide {1} -WAIT -HOST "cu0{2}:28"'.format(self.dst_path,
                                                                                            self.glide_in, cpu_id)
        os.system(cmd)

class Ifp():
    def __init__(self, protein_file, lig_path, csv_file, ifp_type='plec'):
        self.protein = next(oddt.toolkit.readfile('pdb', protein_file))
        self.protein.protein = True
        self.lig_path = lig_path
        self.csv_file = csv_file
        self.ifp_type = ifp_type

    def cal_ifp(self, ligand_name):
        ligand_file = f'{self.lig_path}/{ligand_name}.sdf'
        try:
            # 读取小分子
            ligand = next(oddt.toolkit.readfile('sdf', ligand_file))
            # 计算FP
            # 判断指纹类型并计算
            if self.ifp_type == 'splif':
                fp = fingerprints.SPLIF(ligand, self.protein, depth=2, size=2048, distance_cutoff=4.5)
                fp = fingerprints.sparse_to_dense(fp, size=2048).tolist()  # 转换指纹长度至1024
            elif self.ifp_type == 'ifp':
                fp = fingerprints.InteractionFingerprint(ligand, self.protein, strict=True).tolist()
            elif self.ifp_type == 'silirid':
                fp = fingerprints.SimpleInteractionFingerprint(ligand, self.protein, strict=True).tolist()
            else:
                fp = fingerprints.PLEC(ligand, self.protein, sparse=False, size=2048).tolist()
            # 合并列表
            result = [ligand_name] + fp
            return result
        except:
            print(f'error for {ligand_name}')

    def cal_ifp_2_csv(self, ligand_name, similarity, label):
        result = self.cal_ifp(ligand_name)
        if result:
            result += [similarity, label]
            # 把结果写入csv
            mycsv = open(self.csv_file, 'a')
            mycsvwriter = csv.writer(mycsv)
            mycsvwriter.writerow(result)
            mycsv.close()


    def merge_ifp_label(self, names, *args):
        df = pd.read_csv(self.csv_file, encoding='utf-8', header=None).dropna()
        df.columns = ['name'] + [i for i in range(1, df.shape[1])]
        df.name = df.name.astype(str)
        df1 = pd.DataFrame([names] + list(args), index=['name', 'mw', 'logp', 'rb', 'hba', 'hbr', 'halx']).T
        df = pd.merge(left=df1, right=df, on='name')
        df.to_csv(self.csv_file, encoding='utf-8', index=False)