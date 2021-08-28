#!/bin/bash
# env
source activate dgl_env
for target in ESR1 FEN1 GBA KAT2A MTORC1 PKM2 TP53 VDR
do
  echo ${target}
  # prepare dataset
  python properties_filter.py --target ${target}
  # grid filter
  python grid_filter.py --target ${target} --grid_unit 1000
  # grid filter
  python grid_filter.py --target ${target} --grid_unit 300
done
