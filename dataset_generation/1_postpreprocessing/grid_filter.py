#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2021/4/12 19:25
# @author : Xujun Zhang
import argparse
from class_base.grid_filter_base import grid_filter

# init
argparser = argparse.ArgumentParser()
argparser.add_argument('--path', type=str, default='/root/zxj/du/total_workflow')
argparser.add_argument('--target', type=str, default='mpro')
argparser.add_argument('--src_path', type=str, default='2_propterties_filter')
argparser.add_argument('--dst_path', type=str, default='2_propterties_filter')
argparser.add_argument('--src_file', type=str, default='property_filtered.csv')
argparser.add_argument('--ecfp_file', type=str, default='ecfp.csv')
argparser.add_argument('--tsne_file', type=str, default='tsne.csv')
argparser.add_argument('--dst_file', type=str, default='filtered.csv')
argparser.add_argument('--decoys_size', type=int, default=50)
argparser.add_argument('--grid_unit', type=int, default=200)
args = argparser.parse_args()
#
path = f'{args.path}/{args.target}'
src_path = f'{path}/{args.src_path}'
dst_path = f'{path}/{args.dst_path}'
src_csv = f'{src_path}/{args.src_file}'
ecfp_csv = f'{dst_path}/{args.ecfp_file}'
tsne_csv = f'{dst_path}/{args.tsne_file}'
dst_csv = f'{dst_path}/{args.grid_unit}_{args.dst_file}'
# read
grid_filter = grid_filter(
    src_csv=src_csv,
    ecfp_csv=ecfp_csv,
    tsne_csv=tsne_csv,
    dst_csv=dst_csv,
    decoys_size=args.decoys_size,
    grid_unit=args.grid_unit
)

