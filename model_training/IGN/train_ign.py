#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2021/3/5 9:50
# @author : Xujun Zhang

import argparse
import gc
import sys
import rdkit
from GraphConstructor_V4 import *
from MyUtils_V4 import *
from MyModel_V4 import *
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import time
from sklearn.metrics import mean_squared_error, roc_auc_score
import warnings
import torch
import torch.multiprocessing as mp
import pandas as pd
import os
from _thread import start_new_thread
from functools import wraps
import traceback
from torch.nn.parallel import DistributedDataParallel
from dgl.data.utils import Subset
from dgl.data import split_dataset
import sys

torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')


# os.chdir('./dpi_v4')


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def run_a_train_epoch(model, loss_fn, train_dataloader, optimizer, device):
    # training model for one epoch
    model.train()
    for i_batch, batch in enumerate(train_dataloader):
        model.zero_grad()
        bg, bg3, labels = batch
        bg, bg3, labels = bg.to(device), bg3.to(device), labels.to(device)
        outputs = model(bg, bg3)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()


def run_a_eval_epoch(model, validation_dataloader, device):
    true = []
    pred = []
    model.eval()
    with torch.no_grad():
        for i_batch, batch in enumerate(validation_dataloader):
            # DTIModel.zero_grad()
            bg, bg3, labels = batch
            bg, bg3, labels = bg.to(device), bg3.to(device), labels.to(device)
            outputs = model(bg, bg3)
            true.append(labels.data.cpu().numpy())
            pred.append(outputs.data.cpu().numpy())
    return true, pred





def run(proc_id, num_devices, args, devices, repetition_th, stopper):
    """
    :param proc_id: process id and gpu id
    :param num_devices:
    :param args:
    :param devices: list obeject, [7,6,5,4]
    :return:
    """
    # init
    # train_result_csv = f'{dst_path}/{target}_{repetition_th}_train_result.csv'
    test_result_csv = f'{dst_path}/{target}_{job_type}_test_result.csv'
    #
    dev_id = devices[proc_id]
    dev_id = torch.device('cuda:' + str(dev_id))
    # if num_devices > 1:
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
    torch.distributed.init_process_group(backend="nccl",
                                             init_method=dist_init_method,
                                             world_size=num_devices,
                                             rank=proc_id)
    torch.cuda.set_device(dev_id)
    # stat_res = []
    set_random_seed(repetition_th)
    # load the corresponding data points assigned for each process
    train_dataset = GraphDatasetVS1MP_zxj(graph_file=train_graph, train_or_val='train', num_devices=num_devices, proc_id=proc_id)
    print('the number of train data:', len(train_dataset))
    train_dataloader = DataLoaderX(train_dataset, args.batch_size, shuffle=True, collate_fn=collate_fn_vs)

    # model
    DTIModel = DTIPredictorV4VS(node_feat_size=args.node_feat_size, edge_feat_size=args.edge_feat_size_3d,
                                num_layers=args.num_layers,
                                graph_feat_size=args.graph_feat_size, outdim_g3=args.outdim_g3,
                                d_FC_layer=args.d_FC_layer, n_FC_layer=args.n_FC_layer, dropout=args.dropout,
                                n_tasks=args.n_tasks)
    DTIModel = DTIModel.to(dev_id)
    print('number of parameters : ', sum(p.numel() for p in DTIModel.parameters() if p.requires_grad))
    # if num_devices > 1:
    DTIModel = DistributedDataParallel(DTIModel, device_ids=[dev_id], output_device=dev_id,
                                           find_unused_parameters=True)
    optimizer = torch.optim.Adam(DTIModel.parameters(), lr=args.lr, weight_decay=args.l2)
    loss_fn = FocalLoss(gamma=2, alpha=decoys_ratio)
    for epoch in range(args.epochs):
        st = time.time()
        # train
        run_a_train_epoch(DTIModel, loss_fn, train_dataloader, optimizer, dev_id)

        # if num_devices > 1:
        torch.distributed.barrier()

        if proc_id == 0:
            valid_dataloader = DataLoaderX(valid_dataset, args.batch_size, collate_fn=collate_fn_vs)
            # test
            train_true, train_pred = run_a_eval_epoch(DTIModel, train_dataloader, dev_id)
            valid_true, valid_pred = run_a_eval_epoch(DTIModel, valid_dataloader, dev_id)

            train_true = np.concatenate(np.array(train_true), 0)
            train_pred = np.concatenate(np.array(train_pred), 0)

            valid_true = np.concatenate(np.array(valid_true), 0)
            valid_pred = np.concatenate(np.array(valid_pred), 0)

            train_loss = loss_fn(torch.tensor(train_pred, dtype=torch.float),
                                 torch.tensor(train_true, dtype=torch.float))
            valid_loss = loss_fn(torch.tensor(valid_pred, dtype=torch.float),
                                 torch.tensor(valid_true, dtype=torch.float))
            # train_auc = roc_auc_score(train_true, train_pred)
            # valid_auc = roc_auc_score(valid_true, valid_pred)

            early_stop = stopper.step(valid_loss, DTIModel)
            end = time.time()
            print("epoch:%s \t train_loss:%.4f \t valid_loss:%.4f \t time:%.3f s" % (
            epoch, train_loss, valid_loss, end - st))
            # print("epoch:%s \t train_auc:%.4f \t valid_auc:%.4f \t time:%.3f s" % (
            # epoch, train_auc, valid_auc, end - st))

            if early_stop:
                # load the best model
                stopper.load_checkpoint(DTIModel)
                # train_true, train_pred = run_a_eval_epoch(DTIModel, train_dataloader, device)
                # valid_true, valid_pred = run_a_eval_epoch(DTIModel, valid_dataloader, device)
                # load all the train data set
                valid_true, valid_pred = run_a_eval_epoch(DTIModel, valid_dataloader, dev_id)

                # metrics
                valid_true = np.concatenate(np.array(valid_true), 0).flatten()
                valid_pred = np.concatenate(np.array(valid_pred), 0).flatten()
                pd_va = pd.DataFrame({'valid_true': valid_true, 'valid_pred': valid_pred})
                pd_va.to_csv(test_result_csv, index=False)
                print('main Early stop!')
                break
        # else:
        #     print(proc_id)
    print(f'{proc_id} end')
    cmd = "ps -ef|grep 'python -u'|awk '{print $2}'|xargs kill -9"
    os.system(cmd)


def thread_wrapped_func(func):
    """
    Wraps a process entry point to make it work with OpenMP.
    """

    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = mp.Queue()

        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)

    return decorated_function


def get_ratio_for_loss(csv_file):
    df = pd.read_csv(csv_file)
    if args.n_tasks > 1:
        df_inac = df[df.iloc[:, 3] == 0]
    else:
        df_inac = df[df.iloc[:, -1] == 0]
    return len(df_inac)/len(df)


argparser = argparse.ArgumentParser()
argparser.add_argument('--gpus', type=str, default='0',
                       help="Comma separated list of GPU device IDs.")
argparser.add_argument('--lr', type=float, default=10 ** -3,
                       help="Learning rate")
argparser.add_argument('--epochs', type=int, default=5000,
                       help="Number of epochs in total")
argparser.add_argument('--batch-size', type=int, default=128,
                       help="Batch size")
argparser.add_argument('--tolerance', type=float, default=0.0)
argparser.add_argument('--patience', type=int, default=10)
argparser.add_argument('--l2', type=float, default=10 ** -6)
argparser.add_argument('--repetitions', type=int, default=2)
argparser.add_argument('--node-feat-size', type=int, default=40)
argparser.add_argument('--edge-feat-size-3d', type=int, default=21)
argparser.add_argument('--graph-feat-size', type=int, default=200)
argparser.add_argument('--num-layers', type=int, default=4)
argparser.add_argument('--outdim-g3', type=int, default=128)
argparser.add_argument('--d-FC-layer', type=int, default=200)
argparser.add_argument('--n-FC-layer', type=int, default=2)
argparser.add_argument('--dropout', type=float, default=0.1)
argparser.add_argument('--n-tasks', type=int, default=1)
argparser.add_argument('--target', type=str)
argparser.add_argument('--job_type', type=str, help='train_300|train_1000|train_top50|train_lit')
args = argparser.parse_args()
target = args.target
job_type = args.job_type
# init
path = r'/apdcephfs/private_xujunzhang/project_5'
graph_path = f'{path}/graph/{target}'
train_graph = f'{graph_path}/{job_type}.pkl'
dst_path = f'{path}/model/{target}'


if __name__ == '__main__':
    model_file = f'{dst_path}/{target}_{job_type}_5a8d.pth'
    if not os.path.exists(model_file):
        os.makedirs(dst_path, exist_ok=True)
        # valida_idxes = np.random.RandomState(seed=42).permutation(valida_idxes)
        # instance
        valid_dataset = GraphDatasetVS1MP_zxj(graph_file=train_graph, train_or_val='val')
        print('the number of valid data:', len(valid_dataset))
        # get ratio
        decoys_ratio = valid_dataset.decoys_ratio  # get_ratio_for_loss(train_csv)
        # change data_loader func
        if args.n_tasks > 1:
            collate_fn_vs = collate_fn_vs_multitask_zxj
        # get device
        devices = list(map(int, args.gpus.split(',')))
        num_devices = len(devices)
        stopper = EarlyStopping(mode='lower', patience=args.patience, tolerance=args.tolerance,
                                filename=model_file)
        if num_devices < 1:
            run(0, num_devices, args, devices, 1, stopper)
        else:
            procs = []
            for proc_id in range(num_devices):
                p = mp.Process(target=thread_wrapped_func(run),
                               args=(proc_id, num_devices, args, devices, 1, stopper))
                p.start()
                procs.append(p)

            for p in procs:
                p.join()
    else:
        print(f'{model_file} exists! skip job type of {job_type}')
