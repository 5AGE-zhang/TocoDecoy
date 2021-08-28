#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2021/3/9 11:16
# @author : Xujun Zhang

import argparse
import pandas as pd
from GraphConstructor_V4 import *
from MyUtils_V4 import *
from MyModel_V4 import *
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

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
            if labels.shape[1] == 5:
                labels = labels.reshape((-1, args.n_tasks)).T[0]
            if outputs.shape[1] == 5:
                outputs = outputs.reshape((-1, args.n_tasks)).T[0]
            true.append(labels.data.cpu().numpy())
            pred.append(outputs.data.cpu().numpy())
    true = np.concatenate(np.array(true), 0).flatten()
    pred = np.concatenate(np.array(pred), 0).flatten()
    return true, pred


# init
path = r'/apdcephfs/private_xujunzhang/project_5'
src_path = f'{path}'
if __name__ == '__main__':
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
    argparser.add_argument('--patience', type=int, default=15)
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
    # load test set
    path = r'/apdcephfs/private_xujunzhang/project_5'
    graph_path = f'{path}/graph/{target}'
    train_graph = f'{graph_path}/{job_type}.pkl'
    dst_path = f'{path}/model/{target}'
    model_file = f'{dst_path}/{target}_{job_type}_5a8d.pth'
    stopper = EarlyStopping(mode='lower', patience=args.patience, tolerance=args.tolerance,
                            filename=model_file)
    my_device = torch.device('cuda:0')
    args.n_tasks = 1
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    torch.distributed.init_process_group(backend="nccl",
                                         init_method=dist_init_method,
                                         world_size=1,
                                         rank=0)
    DTIModel = DTIPredictorV4VS(node_feat_size=args.node_feat_size, edge_feat_size=args.edge_feat_size_3d,
                                num_layers=args.num_layers,
                                graph_feat_size=args.graph_feat_size, outdim_g3=args.outdim_g3,
                                d_FC_layer=args.d_FC_layer, n_FC_layer=args.n_FC_layer, dropout=args.dropout,
                                n_tasks=args.n_tasks)
    DTIModel = DTIModel.to(my_device)

    DTIModel = DistributedDataParallel(DTIModel, device_ids=[my_device], output_device=my_device,
                                       find_unused_parameters=True)
    stopper.load_checkpoint(DTIModel)
    # get index
    for test_type in ['test_lit', 'test_filtered']:
        dst_csv = f'{dst_path}/{target}_{job_type}_{test_type}.csv'
        test_graph = f'{graph_path}/{test_type}.pkl'
        test_dataset = GraphDatasetVS1MP_zxj(graph_file=test_graph, train_or_val='test')
        print('the number of test data:', len(test_dataset))
        test_dataset_dataloader = DataLoaderX(test_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn_vs)
        y_true, y_pred = run_a_eval_epoch(model=DTIModel, validation_dataloader=test_dataset_dataloader, device=my_device)
        pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).to_csv(dst_csv, index=False)
