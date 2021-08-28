#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2021/3/27 13:38
# @author : Xujun Zhang

import os
from multiprocessing import Pool

import pandas as pd

import interaction_base

src_path = '/home/Project_5/mix/aldh1'
protein_file = f'{src_path}/docked/5l2n_for_interaction.pdb'
path_for_lig = f'{src_path}/ligands_sdf'
# path_for_mol = f'{path_for_lig}/ligands_sdf'
dst_path = '/home/Project_5/interaction_frequency/inactive'
path_for_complex = f'{dst_path}/complexs'
source_csv = f'{dst_path}/aldh1_filtered.csv'
interaction_csv = f'{dst_path}/interaction.csv'
data_collect_csv = f'{dst_path}/data_collect.csv'
if not os.path.exists(path_for_complex):
    os.mkdir(path_for_complex)
# 实例化对象
this_interaction = interaction_base.oddt_interaction(protein_file=protein_file,
                                                     path_for_lig=path_for_lig,
                                                     path_for_complex=path_for_complex,
                                                     interaction_csv=interaction_csv,
                                                     data_collect_csv=data_collect_csv)


# 定义函数
def each_lig_interaction(lig_names, out_lis=False):
    ligands = ['{}/{}.sdf'.format(path_for_lig, lig_name) for lig_name in lig_names if
               os.path.exists('{}/{}.sdf'.format(path_for_lig, lig_name))]
    # 组成列表
    length = len(ligands)
    ligands_hb = zip(ligands, ['hb'] * length, [out_lis] * length)
    ligands_clb = zip(ligands, ['clb'] * length, [out_lis] * length)
    ligands_qq = zip(ligands, ['qq'] * length, [out_lis] * length)
    ligands_lipo = zip(ligands, ['lipo'] * length, [out_lis] * length)
    ligands_metal = zip(ligands, ['metal'] * length, [out_lis] * length)
    # 多进程计算 计算H键作用
    pool = Pool(28)
    print('cal Hb')
    hb_infos = pool.starmap(this_interaction.interaction2recNum, ligands_hb)
    print('cal clb')
    clb_infos = pool.starmap(this_interaction.interaction2recNum, ligands_clb)
    print('cal qq')
    qq_infos = pool.starmap(this_interaction.interaction2recNum, ligands_qq)
    print('cal lipo')
    lipo_infos = pool.starmap(this_interaction.interaction2recNum, ligands_lipo)
    print('cal metal')
    metal_infos = pool.starmap(this_interaction.interaction2recNum, ligands_metal)
    pool.close()
    pool.join()
    # 合并数据
    interaction_infos = zip(lig_names, hb_infos, clb_infos, qq_infos, lipo_infos, metal_infos)
    # 写入CSV
    pd.DataFrame(interaction_infos,
                 columns=['name', 'hb', 'halogenbond', 'salt_bridges', 'hydrophobic', 'metal']).to_csv(
        this_interaction.interaction_csv, index=False)


if __name__ == '__main__':
    # 获取分子名
    df = pd.read_csv(source_csv).dropna()
    df = df[df.iloc[:, -1] == 0]  # get active
    lig_names = df.iloc[:, 0].values
    lig_names = list(filter(lambda x: os.path.exists(f'{path_for_lig}/{x}.sdf'), lig_names))
    # generate complex
    print('generating complex')
    pool = Pool(28)
    pool.map(this_interaction.generate_complex, lig_names)
    pool.close()
    pool.join()
    # 计算每个分子的相互作用
    print('cal interactions')
    each_lig_interaction(lig_names)
    # 计算频次,写入CSV  pipeline模式下，相互作用的统计用训练集中的活性分子的数据
    pool = Pool(28)
    pool.map(this_interaction.get_rec_frequence, this_interaction.interactions.keys())
    pool.close()
    pool.join()
