#!usr/bin/env python
# -*- coding:utf-8 -*-
import os, pandas as pd, shutil
import sys
from multiprocessing import Pool


def smina_score(ligand):
    single_ligand = '{}/{}.mol2'.format(lig_path, ligand)
    ligand_pdbqt = '{}/{}.pdbqt'.format(pdbqt_dir, ligand)
    log_file = '{}/{}.txt'.format(log_dir, ligand)  # 分数文件
    # 计算
    if os.path.exists(ligand_pdbqt):
        cmdline = '/home/xujun/Soft/Score_Function/smina/smina --receptor %s --ligand %s ' \
                  '--center_x %s --center_y %s --center_z %s --size_x 18.75 --size_y 18.75 --size_z 18.75 --log %s ' \
                  '--custom_scoring /home/xujun/Soft/Score_Function/smina/total.score ' \
                  '--score_only --cpu 1' % (protein_pred, ligand_pdbqt, x, y, z, log_file)
        os.system(cmdline)
    else:
        cmdline = 'module purge &&'
        cmdline += 'module load autodock &&'
        cmdline += 'prepare_ligand4.py -l %s -o %s -A hydrogens &&' % (single_ligand, ligand_pdbqt)
        cmdline += '/home/xujun/Soft/Score_Function/smina/smina --receptor %s --ligand %s ' \
                   '--center_x %s --center_y %s --center_z %s --size_x 18.75 --size_y 18.75 --size_z 18.75 --log %s ' \
                   '--custom_scoring /home/xujun/Soft/Score_Function/smina/total.score ' \
                   '--score_only --cpu 1' % (protein_pred, ligand_pdbqt, x, y, z, log_file)
        os.system(cmdline)


def get_result(lig_name, similarity, label):
    log_file = '{}/{}.txt'.format(log_dir, lig_name)
    if os.path.exists(log_file):
        # 整合结果
        with open(log_file, 'r') as f:
            con = f.readlines()
        result = [lig_name]
        for i in range(len(con)):
            if con[i].startswith('Term values, before weighting:'):
                result = [lig_name] + [x.strip() for x in con[i + 1].lstrip('##  ').split(' ')] + [similarity, label]
        pd.DataFrame(result).T.to_csv(csv_file, index=False, header=False, mode='a')


if __name__ == '__main__':
    # 定义文件
    job_type = 'aldh1'  # fpb pcb
    path = '/home/xujun/Project_5'
    # protein and ligand path
    src_path = f'{path}/mix/{job_type}'
    src_pro_path = f'{src_path}/docked'
    protein_file = f'{src_pro_path}/5l2n_protein.pdb'
    crystal_file = f'{src_pro_path}/reference.mol2'
    lig_path = f'{src_path}/ligands_mol2'
    log_dir = f'{src_path}/smina_log'  # 分数日志文件夹
    # descriptors dir
    dst_path = f'/home/xujun/Project_5/cal_descriptors/v_1/{job_type}'
    pdbqt_dir = '{}/pdbqt'.format(src_path)  # pdbqt文件夹
    csv_file = f'{dst_path}/smina.csv'
    # get lig_names
    src_csv = f'{dst_path}/{job_type}_filtered.csv'
    docked_csv = f'{dst_path}/SP_ifp.csv'
    df_sp = pd.read_csv(docked_csv, encoding='utf-8').dropna()
    names = df_sp.iloc[:, 0].values
    df = pd.read_csv(src_csv, encoding='utf-8')
    df.index = df.iloc[:, 0].values
    similarities = df.loc[names, 'train'].values  # similarity
    labels = df.loc[names, 'label'].values
    # smina
    protein_pre = f'{src_pro_path}/5l2n_smina_p.pdb'
    protein_pred = f'{src_pro_path}/5l2n_smina_p.pdbqt'
    crystal_pdbqt = f'{src_pro_path}/5l2n_crystal_ligand.pdbqt'
    # 移动文件并转换格式
    if not os.path.exists(crystal_pdbqt):
        cmdline = 'module purge &&'
        cmdline += 'module load autodock &&'
        cmdline += 'prepare_ligand4.py -l {} -o {} -A hydrogens '.format(crystal_file, crystal_pdbqt)
        os.system(cmdline)
    # 获取 口袋坐标
    x = os.popen(
        "cat %s | awk '{if ($1==\"ATOM\") print $(NF-6)}' | awk '{x+=$1} END {print x/(NR)}'" % crystal_pdbqt).read()
    y = os.popen(
        "cat %s | awk '{if ($1==\"ATOM\") print $(NF-5)}' | awk '{y+=$1} END {print y/(NR)}'" % crystal_pdbqt).read()
    z = os.popen(
        "cat %s | awk '{if ($1==\"ATOM\") print $(NF-4)}' | awk '{z+=$1} END {print z/(NR)}'" % crystal_pdbqt).read()
    x = float(x.strip())
    y = float(y.strip())
    z = float(z.strip())
    # 处理蛋白
    if not os.path.exists(protein_pred):
        cmdline = 'cat %s | sed \'/HETATM/\'d > %s &&' % (protein_file, protein_pre)
        cmdline += 'module purge &&'
        cmdline += 'module load autodock &&'
        cmdline += 'prepare_receptor4.py -r %s -o %s -A hydrogens -U nphs_lps_waters_nonstdres &&' % (
            protein_pre, protein_pred)
        cmdline += 'rm  %s' % protein_pre
        os.system(cmdline)
    # 创建文件夹
    if not os.path.exists(pdbqt_dir):
        os.mkdir(pdbqt_dir)
    # 创建CSV
    if not os.path.exists(csv_file):
        pd.DataFrame(
        ['name', 'gauss(o=0,_w=0.3,_c=8)', 'gauss(o=0.5,_w=0.3,_c=8)', 'gauss(o=1,_w=0.3,_c=8)',
         'gauss(o=1.5,_w=0.3,_c=8)', 'gauss(o=2,_w=0.3,_c=8)', 'gauss(o=2.5,_w=0.3,_c=8)',
         'gauss(o=0,_w=0.5,_c=8)'
            , 'gauss(o=1,_w=0.5,_c=8)', 'gauss(o=2,_w=0.5,_c=8)', 'gauss(o=0,_w=0.7,_c=8)',
         'gauss(o=1,_w=0.7,_c=8)',
         'gauss(o=2,_w=0.7,_c=8)', 'gauss(o=0,_w=0.9,_c=8)', 'gauss(o=1,_w=0.9,_c=8)', 'gauss(o=2,_w=0.9,_c=8)',
         'gauss(o=3,_w=0.9,_c=8)', 'gauss(o=0,_w=1.5,_c=8)', 'gauss(o=1,_w=1.5,_c=8)', 'gauss(o=2,_w=1.5,_c=8)',
         'gauss(o=3,_w=1.5,_c=8)', 'gauss(o=4,_w=1.5,_c=8)', 'gauss(o=0,_w=2,_c=8)', 'gauss(o=1,_w=2,_c=8)',
         'gauss(o=2,_w=2,_c=8)', 'gauss(o=3,_w=2,_c=8)', 'gauss(o=4,_w=2,_c=8)', 'gauss(o=0,_w=3,_c=8)',
         'gauss(o=1,_w=3,_c=8)', 'gauss(o=2,_w=3,_c=8)', 'gauss(o=3,_w=3,_c=8)', 'gauss(o=4,_w=3,_c=8)',
         'repulsion(o=0.4,_c=8)', 'repulsion(o=0.2,_c=8)', 'repulsion(o=0,_c=8)', 'repulsion(o=-0.2,_c=8)',
         'repulsion(o=-0.4,_c=8)', 'repulsion(o=-0.6,_c=8)', 'repulsion(o=-0.8,_c=8)', 'repulsion(o=-1,_c=8)',
         'hydrophobic(g=0.5,_b=1.5,_c=8)', 'hydrophobic(g=0.5,_b=1,_c=8)', 'hydrophobic(g=0.5,_b=2,_c=8)',
         'hydrophobic(g=0.5,_b=3,_c=8)', 'non_hydrophobic(g=0.5,_b=1.5,_c=8)', 'vdw(i=4,_j=8,_s=0,_^=100,_c=8)',
         'vdw(i=6,_j=12,_s=1,_^=100,_c=8)', 'non_dir_h_bond(g=-0.7,_b=0,_c=8)',
         'non_dir_h_bond(g=-0.7,_b=0.2,_c=8)'
            , 'non_dir_h_bond(g=-0.7,_b=0.5,_c=8)', 'non_dir_h_bond(g=-1,_b=0,_c=8)',
         'non_dir_h_bond(g=-1,_b=0.2,_c=8)',
         'non_dir_h_bond(g=-1,_b=0.5,_c=8)', 'non_dir_h_bond(g=-1.3,_b=0,_c=8)',
         'non_dir_h_bond(g=-1.3,_b=0.2,_c=8)',
         'non_dir_h_bond(g=-1.3,_b=0.5,_c=8)', 'non_dir_anti_h_bond_quadratic(o=0,_c=8)',
         'non_dir_anti_h_bond_quadratic(o=0.5,_c=8)', 'non_dir_anti_h_bond_quadratic(o=1,_c=8)',
         'donor_donor_quadratic(o=0,_c=8)', 'donor_donor_quadratic(o=0.5,_c=8)',
         'donor_donor_quadratic(o=1,_c=8)',
         'acceptor_acceptor_quadratic(o=0,_c=8)', 'acceptor_acceptor_quadratic(o=0.5,_c=8)',
         'acceptor_acceptor_quadratic(o=1,_c=8)', 'non_dir_h_bond_lj(o=-0.7,_^=100,_c=8)',
         'non_dir_h_bond_lj(o=-1,_^=100,_c=8)', 'non_dir_h_bond_lj(o=-1.3,_^=100,_c=8)',
         'ad4_solvation(d-sigma=3.6,_s/q=0.01097,_c=8)', 'ad4_solvation(d-sigma=3.6,_s/q=0,_c=8)',
         'electrostatic(i=1,_^=100,_c=8)', 'electrostatic(i=2,_^=100,_c=8)', 'num_tors_div',
         'num_tors_div_simple',
         'num_heavy_atoms_div', 'num_heavy_atoms', 'num_tors_add', 'num_tors_sqr', 'num_tors_sqrt',
         'num_hydrophobic_atoms', 'ligand_length', 'num_ligands', 'similarity', 'label']).T.to_csv(
        csv_file, index=False, header=False)
    # 创建pdbqt文件夹
    if not os.path.exists(pdbqt_dir):
        os.mkdir(pdbqt_dir)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)
    else:
        os.mkdir(log_dir)
    # 调用软件
    pool = Pool(28)
    pool.map(smina_score, names)
    pool.close()
    pool.join()
    # 汇总分数
    pool = Pool(28)
    pool.starmap(get_result, zip(names, similarities, labels))
    pool.close()
    pool.join()


