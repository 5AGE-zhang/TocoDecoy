#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2021/1/31 18:19
# @author : Xujun Zhang

import os

import ddc_pub.ddc_v3 as ddc
from my_utils import load_dataset, mol_controller
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
# set GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"  # 使用第二块GPU（从0开始）
# init
path = os.path.dirname(os.path.realpath(__file__))
dataset_path = f'{path}/datasets'
model_path = f'{path}/models/'
train_dataset = f'{dataset_path}/CHEMBL25_TRAIN_MOLS.h5'
model_name = 'my_pcb_model'
# read_data_set
print('load training data....')
train_mols = load_dataset(train_dataset)
# Get the SMILES behind the binary mols
print('calculate properties')
moler = mol_controller(mols=train_mols)
train_x = moler.cal_properties()
# vocab_dic
# All apriori known characters of the SMILES in the dataset
charset = "Brc1(-23[nH])45C=NOso#FlS67+89%0"
# Apriori known max length of the SMILES in the dataset
maxlen = 128
# Name of the dataset
name = "ChEMBL25_TRAIN"
dataset_info = {"charset": charset, "maxlen": maxlen, "name": name}
# model
print('build model')
model = ddc.DDC(x=train_x,
                y=train_mols,
                scaling=True,
                pca=False,
                dataset_info=dataset_info,
                noise_std=0.1,
                lstm_dim=256,
                dec_layers=3,
                td_dense_dim=0,
                batch_size=128)
# train
print('start training....')
model.fit(
    epochs=100,
    lr=1e-3,
    mini_epochs=10,
    patience=25,
    model_name=model_name,
    gpus=1,
    workers=1,
    use_multiprocessing=False,
    verbose=2,
    max_queue_size=10,
    clipvalue=0,
    save_period=5,
    checkpoint_dir=model_path,
    lr_decay=True,
    sch_epoch_to_start=500,
    sch_last_epoch=999,
    sch_lr_init=1e-3,
    sch_lr_final=1e-6,
)
# Save the final model
model.save(model_name)