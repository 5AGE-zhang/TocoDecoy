#!usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, glob
import pandas as pd
import shutil
from multiprocessing import Pool
from functools import partial
from NNScore2module import PDB, binana, command_line_parameters


# def nn_score(lig_name, similarity, label, ligand_path, csv_file):
def nn_score(zip_lis, ligand_path, csv_file):
    lig_name, similarity, label = zip_lis[0], zip_lis[1], zip_lis[2]
    import time
    ligand_pdbqt = '%s/%s.pdbqt' % (ligand_path, lig_name)
    ligand_pdbqt_pred = '%s/%s_pre.pdbqt' % (ligand_path, lig_name)
    # 准备小分子
    # log_file = '%s/%s_nn.txt' % (log_dir, lig_name)  # 分数文件
    if not os.path.exists(ligand_pdbqt):
        return None
    # 额外处理
    if not os.path.exists(ligand_pdbqt_pred):
        with open(ligand_pdbqt, 'r')as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            if line.startswith('ATOM'):
                new_lines.append(line[:23] + '   ' + line[26:])
            else:
                new_lines.append(line)
        new_lig = ''.join(new_lines)
        with open(ligand_pdbqt_pred, 'w')as f:
            f.write(new_lig)
    # 计算
    cmd = "/home/xujun/Soft/Score_Function/NNscore/NNScore2module.py -receptor %s -ligand %s" % (
    protein_pred, ligand_pdbqt_pred)
    try:
        params_list = cmd.split()
        cmd_params = command_line_parameters(params_list)
        receptor = PDB()
        receptor.LoadPDB_from_file(protein_pred)
        receptor.OrigFileName = protein_pred
        d = binana(ligand_pdbqt_pred, receptor, cmd_params, "", "", "")

        result = [
                     lig_name] + d.vina_output + d.ligand_receptor_atom_type_pairs_less_than_two_half.values() + d.ligand_receptor_atom_type_pairs_less_than_four.values() \
                 + d.ligand_atom_types.values() + d.ligand_receptor_atom_type_pairs_electrostatic.values() + d.rotateable_bonds_count.values() \
                 + d.active_site_flexibility.values() + d.hbonds.values() + d.hydrophobics.values() + d.stacking.values() + d.pi_cation.values() \
                 + d.t_shaped.values() + d.salt_bridges.values() + [similarity, label]
    except:
        result = [lig_name]
    # with open(log_file, 'w')as f:
    #     f.write(str(result))
    # # 整合结果
    # with open(log_file, 'r')as f:
    #     lines = f.readlines()
    # result = eval(lines[0].strip())
    pd.DataFrame(result).T.to_csv(csv_file, index=None, header=None, mode='a')
    # # 删除分数文件
    # if len(result) != 1:
    #     os.remove(ligand_pdbqt_pred)


if __name__ == '__main__':
    # 定义不同原子类型
    vina_output_list = ['vina_affinity', 'vina_gauss_1', 'vina_gauss_2', 'vina_repulsion', 'vina_hydrophobic',
                        'vina_hydrogen']
    ligand_receptor_atom_type_pairs_less_than_two_half_list = ['A_MN', 'OA_SA', 'HD_N', 'N_ZN', 'A_MG', 'HD_NA', 'A_CL',
                                                               'MG_OA', 'FE_HD', 'A_OA', 'NA_ZN', 'A_N', 'C_OA', 'F_HD',
                                                               'C_HD', 'NA_SA', 'A_ZN', 'C_NA', 'N_N', 'MN_N', 'F_N',
                                                               'FE_OA', 'HD_I', 'BR_C', 'MG_NA', 'C_ZN', 'CL_MG',
                                                               'BR_OA',
                                                               'A_FE', 'CL_OA', 'CL_N', 'NA_OA', 'F_ZN', 'HD_P',
                                                               'CL_ZN',
                                                               'C_C', 'C_CL', 'FE_N', 'HD_S', 'HD_MG', 'C_F', 'A_NA',
                                                               'BR_HD', 'HD_OA', 'HD_MN', 'A_SA', 'A_F', 'HD_SA', 'A_C',
                                                               'A_A', 'F_SA', 'C_N', 'HD_ZN', 'OA_OA', 'N_SA', 'CL_FE',
                                                               'C_MN', 'CL_HD', 'OA_ZN', 'MN_OA', 'C_MG', 'F_OA',
                                                               'CD_OA',
                                                               'S_ZN', 'N_OA', 'C_SA', 'N_NA', 'A_HD', 'HD_HD', 'SA_ZN']
    ligand_receptor_atom_type_pairs_less_than_four_list = ['I_N', 'OA_SA', 'FE_NA', 'HD_NA', 'A_CL', 'MG_SA', 'A_CU',
                                                           'P_SA', 'C_NA', 'MN_NA', 'F_N', 'HD_N', 'HD_I', 'CL_MG',
                                                           'HD_S',
                                                           'CL_MN', 'F_OA', 'HD_OA', 'F_HD', 'A_SA', 'A_BR', 'BR_HD',
                                                           'SA_SA', 'A_MN', 'N_ZN', 'A_MG', 'I_OA', 'C_C', 'N_S', 'N_N',
                                                           'FE_N', 'NA_SA', 'BR_N', 'MN_N', 'A_P', 'BR_C', 'A_FE',
                                                           'MN_P',
                                                           'CL_OA', 'CU_HD', 'MN_S', 'A_S', 'FE_OA', 'NA_ZN', 'P_ZN',
                                                           'A_F',
                                                           'A_C', 'A_A', 'A_N', 'HD_MN', 'A_I', 'N_SA', 'C_OA', 'MG_P',
                                                           'BR_SA', 'CU_N', 'MN_OA', 'MG_N', 'HD_HD', 'C_FE', 'CL_NA',
                                                           'MG_OA', 'A_OA', 'CL_ZN', 'BR_OA', 'HD_ZN', 'HD_P', 'OA_P',
                                                           'OA_S', 'N_P', 'A_NA', 'CL_FE', 'HD_SA', 'C_MN', 'CL_HD',
                                                           'C_MG',
                                                           'FE_HD', 'MG_S', 'NA_S', 'NA_P', 'FE_SA', 'P_S', 'C_HD',
                                                           'A_ZN',
                                                           'CL_P', 'S_SA', 'CL_S', 'OA_ZN', 'N_NA', 'MN_SA', 'CL_N',
                                                           'NA_OA', 'C_ZN', 'C_CD', 'HD_MG', 'C_F', 'C_I', 'C_CL',
                                                           'C_N',
                                                           'C_P', 'C_S', 'A_HD', 'F_SA', 'MG_NA', 'OA_OA', 'CL_SA',
                                                           'S_ZN',
                                                           'N_OA', 'C_SA', 'SA_ZN']
    ligand_atom_types_list = ['A', 'C', 'CL', 'I', 'N', 'P', 'S', 'BR', 'HD', 'NA', 'F', 'OA', 'SA']
    ligand_receptor_atom_type_pairs_electrostatic_list = ['I_N', 'OA_SA', 'FE_NA', 'HD_NA', 'A_CL', 'MG_SA', 'P_SA',
                                                          'C_NA',
                                                          'MN_NA', 'F_N', 'HD_N', 'HD_I', 'CL_MG', 'HD_S', 'CL_MN',
                                                          'F_OA',
                                                          'HD_OA', 'F_HD', 'A_SA', 'A_BR', 'BR_HD', 'SA_SA', 'A_MN',
                                                          'N_ZN',
                                                          'A_MG', 'I_OA', 'C_C', 'N_S', 'N_N', 'FE_N', 'NA_SA', 'BR_N',
                                                          'MN_N', 'A_P', 'BR_C', 'A_FE', 'MN_P', 'CL_OA', 'CU_HD',
                                                          'MN_S',
                                                          'A_S', 'FE_OA', 'NA_ZN', 'P_ZN', 'A_F', 'A_C', 'A_A', 'A_N',
                                                          'HD_MN', 'A_I', 'N_SA', 'C_OA', 'MG_P', 'BR_SA', 'CU_N',
                                                          'MN_OA',
                                                          'MG_N', 'HD_HD', 'C_FE', 'CL_NA', 'MG_OA', 'A_OA', 'CL_ZN',
                                                          'BR_OA', 'HD_ZN', 'HD_P', 'OA_P', 'OA_S', 'N_P', 'A_NA',
                                                          'CL_FE',
                                                          'HD_SA', 'C_MN', 'CL_HD', 'C_MG', 'FE_HD', 'MG_S', 'NA_S',
                                                          'NA_P',
                                                          'FE_SA', 'P_S', 'C_HD', 'A_ZN', 'CL_P', 'S_SA', 'CL_S',
                                                          'OA_ZN',
                                                          'N_NA', 'MN_SA', 'CL_N', 'NA_OA', 'F_ZN', 'C_ZN', 'HD_MG',
                                                          'C_F',
                                                          'C_I', 'C_CL', 'C_N', 'C_P', 'C_S', 'A_HD', 'F_SA', 'MG_NA',
                                                          'OA_OA', 'CL_SA', 'S_ZN', 'N_OA', 'C_SA', 'SA_ZN']
    rotateable_bonds_count_list = ['rot_bonds']
    active_site_flexibility_list = ['SIDECHAIN_OTHER', 'SIDECHAIN_ALPHA', 'BACKBONE_ALPHA', 'SIDECHAIN_BETA',
                                    'BACKBONE_BETA', 'BACKBONE_OTHER']
    hbonds_list = ['HDONOR-LIGAND_SIDECHAIN_BETA', 'HDONOR-LIGAND_BACKBONE_OTHER', 'HDONOR-LIGAND_SIDECHAIN_ALPHA',
                   'HDONOR-RECEPTOR_SIDECHAIN_OTHER', 'HDONOR-RECEPTOR_BACKBONE_ALPHA',
                   'HDONOR-RECEPTOR_SIDECHAIN_BETA',
                   'HDONOR-RECEPTOR_SIDECHAIN_ALPHA', 'HDONOR-LIGAND_SIDECHAIN_OTHER', 'HDONOR-LIGAND_BACKBONE_BETA',
                   'HDONOR-RECEPTOR_BACKBONE_BETA', 'HDONOR-RECEPTOR_BACKBONE_OTHER', 'HDONOR-LIGAND_BACKBONE_ALPHA']
    hydrophobics_list = ['SIDECHAIN_OTHER', 'SIDECHAIN_ALPHA', 'BACKBONE_ALPHA', 'SIDECHAIN_BETA', 'BACKBONE_BETA',
                         'BACKBONE_OTHER']
    stacking_list = ['ALPHA', 'BETA', 'OTHER']
    pi_cation_list = ['LIGAND-CHARGED_BETA', 'LIGAND-CHARGED_ALPHA', 'RECEPTOR-CHARGED_BETA', 'RECEPTOR-CHARGED_OTHER',
                      'RECEPTOR-CHARGED_ALPHA', 'LIGAND-CHARGED_OTHER']
    t_shaped_list = ['ALPHA', 'BETA', 'OTHER']
    salt_bridges_list = ['ALPHA', 'BETA', 'OTHER']
    # 将不同原子类型组合成列名
    header_list = ['name'] + vina_output_list + ['atp2_%s' % it for it in ligand_receptor_atom_type_pairs_less_than_two_half_list] \
                  + ['atp4_%s' % it for it in ligand_receptor_atom_type_pairs_less_than_four_list] + ['lat_%s' % it for
                                                                                                      it in
                                                                                                      ligand_atom_types_list] \
                  + ['ele_%s' % it for it in
                     ligand_receptor_atom_type_pairs_electrostatic_list] + rotateable_bonds_count_list + [
                      'siteflex_%s' % it for it in active_site_flexibility_list] \
                  + ['hbond_%s' % it for it in hbonds_list] + ['hydrophobic_%s' % it for it in hydrophobics_list] + [
                      'stacking_%s' % it for it in stacking_list] \
                  + ['pi_cation_%s' % it for it in pi_cation_list] + ['t_shaped_%s' % it for it in t_shaped_list] + [
                      'salt_bridges_%s' % it for it in salt_bridges_list] + ['similarity', 'label']

    # 定义文件
    job_type = 'aldh1'  # fpb pcb
    path = '/home/xujun/Project_5'
    # protein and ligand path
    src_path = '{0}/mix/{1}'.format(path, job_type)
    src_pro_path = '{0}/docked'.format(src_path)
    protein_pred = '{0}/5l2n_smina_p.pdbqt'.format(src_pro_path)
    crystal_pdbqt = '{0}/5l2n_crystal_ligand.pdbqt'.format(src_pro_path)
    lig_path = '{0}/ligands_mol2'.format(src_path)
    log_dir = '{0}/nn_log'.format(src_path)  # 分数日志文件夹
    # descriptors dir
    dst_path = '/home/xujun/Project_5/cal_descriptors/v_1/{}'.format(job_type)
    pdbqt_dir = '{0}/pdbqt'.format(src_path)  # pdbqt文件夹
    csv_file = '{0}/nn.csv'.format(dst_path)
    # get lig_names
    src_csv = '{0}/{1}_filtered.csv'.format(dst_path, job_type)
    docked_csv = '{0}/SP.csv'.format(dst_path)
    df_sp = pd.read_csv(docked_csv, encoding='utf-8').dropna()
    names = df_sp.iloc[:, 0].values
    df = pd.read_csv(src_csv, encoding='utf-8')
    df.index = df.iloc[:, 0].values
    similarities = df.loc[names, 'train'].values  # similarity
    labels = df.loc[names, 'label'].values
    # mkdir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    # write csv
    if not os.path.exists(csv_file):
        pd.DataFrame(header_list).T.to_csv(csv_file, index=False, header=False)
    # partial
    nn_ = partial(nn_score, ligand_path=pdbqt_dir, csv_file=csv_file)
    # multiprocessing
    pool = Pool(28)
    pool.map(nn_, zip(names, similarities, labels))
    pool.close()
    pool.join()
