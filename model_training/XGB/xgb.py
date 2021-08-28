#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2021/2/19 16:24
# @author : Xujun Zhang

import os
import pickle
import warnings
from functools import reduce, partial
# from pathos.multiprocessing import ProcessPool as Pool
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')


class xgb_model():
    def __init__(self, model_file, ifps, labels):
        # get file
        self.model_file = model_file
        # train model
        if not os.path.exists(self.model_file):
            print('model file not exist')
            # train
            print('start training......')
            clf = self.train_xgb(ifps, labels)
            # save
            self.save_model(clf)
        # load model
        print('load model......')
        self.model = self.load_model()

    def train_xgb(self, ifps, labels, hyper_rounds=20):
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from xgboost import XGBClassifier
        from hyperopt import hp, fmin, tpe
        # 超参数寻优
        def model(hyper_parameter):  # 待寻优函数
            clf = XGBClassifier(**hyper_parameter,
                                n_jobs=28,
                                random_state=42)
            e = cross_val_score(clf, ifps, labels, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                n_jobs=1,
                                scoring='f1').mean()
            print(f'Mean F1 socre:{e:.3f}')
            return -e

        hyper_parameter = {'n_estimators': hp.choice('n_estimators', range(100, 301, 10)),
                           'max_depth': hp.choice('max_depth', range(3, 11)),
                           'learning_rate': hp.loguniform('learning_rate', 1e-8, 0.1),
                           'reg_lambda': hp.loguniform('reg_lambda', 0.5, 3)}  # 选择要优化的超参数
        # 创建对应的超参数列表
        estimators = [i for i in range(100, 301, 10)]
        depth = [i for i in range(3, 11)]
        # 寻优
        best = fmin(model, hyper_parameter, algo=tpe.suggest, max_evals=hyper_rounds,
                    rstate=np.random.RandomState(42))  # 寻找model()函数的最小值，计算次数为100次
        # 训练
        clf = XGBClassifier(n_estimators=estimators[best['n_estimators']],
                            max_depth=depth[best['max_depth']],
                            learning_rate=best['learning_rate'],
                            reg_lambda=best['reg_lambda'],
                            n_jobs=-1, random_state=42)  # 定义分类器
        clf.fit(X=ifps, y=labels)
        return clf

    def save_model(self, clf):
        with open(self.model_file, 'wb') as f:
            pickle.dump(clf, f)

    def load_model(self):
        with open(self.model_file, 'rb') as f:
            return pickle.load(f)

    def predict(self, x):
        model = self.load_model()
        pred_y = model.predict(x)
        pred_proba = model.predict_proba(x)
        return pred_y, pred_proba.T[1]

    def metric(self, pred_proba, pred_y, y_true):
        from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score, precision_score, confusion_matrix
        f1_ = f1_score(pred_y, y_true)
        acc = accuracy_score(pred_y, y_true)
        roc_auc_ = roc_auc_score(y_true, pred_proba)
        recall = recall_score(y_true=y_true, y_pred=pred_y)
        precision = precision_score(y_true=y_true, y_pred=pred_y)
        tn, fp, fn, tp = confusion_matrix(y_true, pred_y).ravel()
        print(
            f'---------------metric---------------\nF1_score:{f1_:.3f} || Accuracy:{acc:.3f} || ROC_AUC:{roc_auc_:.3f}|| Recall:{recall:.3f} || Precision:{precision:.3f}\nTN:{tn:.3f} || FP:{fp:.3f} || FN:{fn:.3f}|| TP:{tp:.3f}')


def get_des_label(csv_file, des_type):
    # read_file
    df = pd.read_csv(csv_file, encoding='utf-8').dropna()
    # add label
    df_ac = df[df.iloc[:, 0].str.contains('_0')]
    df_ac['label'] = np.ones((len(df_ac)))
    df_inac = df[~df.iloc[:, 0].str.contains('_0')]
    df_inac['label'] = np.zeros((len(df_inac)))
    # merge
    df = df_ac.append(df_inac, sort=False)
    # get data
    if des_type == 'sp':
        df = df.iloc[:, [0] + list(range(7, 19))]
    return df


def get_top_n(molecule_name, df, top_n=50, ascending=True):
    df = df[df.iloc[:, 0].str.startswith(f'{molecule_name}_')]
    df_seed = pd.DataFrame(df.iloc[0, :]).T
    df_decoys = df.iloc[1:, :]
    df_decoys.sort_values(by='similarity', inplace=True, ascending=ascending)
    df_decoys = df_decoys.iloc[:top_n, :]
    df = df_decoys.append(df_seed, sort=False)
    return df


def merge_2_df(df1, df2):
    new_df = df1.append(df2, sort=False)
    return new_df


def collate_csv(csv_file, job_type, top_n):
    # read
    df = pd.read_csv(csv_file, encoding='utf-8').dropna().iloc[:, :]
    if top_n:
        # get_uniqe_names
        df_seed = df[df.iloc[:, 0].str.contains('_0')]
        names = [int(i[0].split('_')[0]) for i in df_seed.values]
        # define
        if job_type == 'fpb':
            get_top_n_ = partial(get_top_n, df=df, top_n=top_n, ascending=False)
        else:
            get_top_n_ = partial(get_top_n, df=df, top_n=top_n, ascending=True)
        # multi
        pool = Pool(28)
        dfs = pool.map(get_top_n_, names)
        pool.close()
        pool.join()
        # merge
        top_n_df = reduce(merge_2_df, dfs)
        top_n_df.columns = df.columns
    else:
        top_n_df = df
    return top_n_df


def train_a_xgb():
    # init
    path = r'/home/xujun/Project_5'
    des_type = 'ecfp_sifp'  # 'ecfp'  SP plec smina_nn
    like_dude = True
    top_n = 0  # 50
    data_path = f'{path}/cal_descriptors/v_1'
    path_model = f'{path}/model/{des_type}'
    # file
    csv_file = f'{path_model}/{des_type}.csv'
    labeled_csv_file = f'{path_model}/{des_type}_top{top_n}.csv'
    model_file = f'{path_model}/xgb_{des_type}_py36.pkl'
    # read df
    if os.path.exists(labeled_csv_file):
        df = pd.read_csv(labeled_csv_file, encoding='utf-8').dropna()
    else:
        # fpb
        print('preprocess fpc csv')
        job_type = 'fpb'
        data_csv = f'{data_path}/{job_type}/{des_type}.csv'
        df_fpb = collate_csv(data_csv, job_type=job_type, top_n=top_n)
        # pcb
        print('preprocess pcb csv')
        job_type = 'pcb'
        data_csv = f'{data_path}/{job_type}/{des_type}.csv'
        df_pcb = collate_csv(data_csv, job_type=job_type, top_n=top_n)
        # merge
        print('merge fpc and pcb csv')
        df = merge_2_df(df_fpb, df_pcb)
        df_seed = df[df.iloc[:, -1].astype(int) == 1]
        df_decoys = df[df.iloc[:, -1].astype(int) == 0]
        try:
            df_seed.drop_duplicates(subset='name', keep='first', inplace=True)
        except:
            df_seed.drop_duplicates(subset='NAME', keep='first', inplace=True)
        df = merge_2_df(df_seed, df_decoys)
        # if des_type == 'SP':
        #     df = df.iloc[:, [0] + list(range(7, 20))]
        df.to_csv(labeled_csv_file, encoding='utf-8', index=False)
    if des_type == 'SP' or des_type == 'smina_nn':
        print('data preprocess')
        # preprocess
        scaler = MinMaxScaler()
        des = df.iloc[:, 1:-2].values
        des = scaler.fit_transform(des)
        df_des = pd.DataFrame(des, columns=df.iloc[:, 1:-2].columns)
        df = pd.concat(
            [pd.DataFrame(df.iloc[:, 0].reset_index(drop=True)), df_des, df.iloc[:, -2:].reset_index(drop=True)],
            axis=1)
        # df.reset_index(drop=True, inplace=True)
    # train_test_split
    train_df, test_df = train_test_split(df, train_size=0.8, shuffle=True, stratify=df.iloc[:, -1].values,
                                         random_state=42)
    if like_dude:
        # train
        df_seed = train_df[train_df.iloc[:, -1].astype(int) == 1]
        df_decoys = train_df[train_df.iloc[:, -2] <= 0.4]
        train_df = merge_2_df(df_seed, df_decoys)
        # test
        df_seed = test_df[test_df.iloc[:, -1].astype(int) == 1]
        df_decoys = test_df[test_df.iloc[:, -2] <= 0.4]
        test_df = merge_2_df(df_seed, df_decoys)
        model_file = f'{path_model}/xgb_{des_type}_dude_py36.pkl'
    # training
    labels = train_df.iloc[:, -1].values.astype(int)
    ifps = train_df.iloc[:, 1:-2].values
    xgb = xgb_model(model_file=model_file, ifps=ifps, labels=labels)
    # validation
    labels = test_df.iloc[:, -1].values.astype(int)
    ifps = test_df.iloc[:, 1:-2].values
    y_pred, y_pred_proba = xgb.predict(ifps)
    # metric
    xgb.metric(pred_proba=y_pred_proba, pred_y=y_pred, y_true=labels)


def valida():
    # validation
    # init
    path = r'/home/xujun/Project_5'
    des_type = 'ecfp'  # 'ecfp'  SP plec
    top_n = 0  # 50

    # strict
    # file
    path_model = f'{path}/model/{des_type}'
    labeled_csv_file = f'{path_model}/{des_type}_top{top_n}.csv'
    model_file = f'{path_model}/xgb_{des_type}.pkl'
    model_strict = xgb_model(model_file=model_file, ifps='', labels='')
    df_strict = pd.read_csv(labeled_csv_file, encoding='utf-8').dropna()
    _, strict_test_df = train_test_split(df_strict, train_size=0.8, shuffle=True, stratify=df_strict.iloc[:, -1].values,
                                         random_state=42)
    labels_strict = strict_test_df.iloc[:, -1].values
    ifps_strict = strict_test_df.iloc[:, 1:-2].values

    # dude
    df_seed = strict_test_df[strict_test_df.iloc[:, -1].astype(int) == 1]
    df_decoys = strict_test_df[strict_test_df.iloc[:, -2] <= 0.4]
    dude_test_df = merge_2_df(df_seed, df_decoys)
    model_file = f'{path_model}/xgb_{des_type}_dude.pkl'
    model_dude = xgb_model(model_file=model_file, ifps='', labels='')
    labels_dude = dude_test_df.iloc[:, -1].values
    ifps_dude = dude_test_df.iloc[:, 1:-2].values

    # LIT-PCBA
    target = 'aldh1'
    des_type = 'ecfp'  # 'ecfp'  SP plec smina_nn
    data_path = f'{path}/cal_descriptors/v_1/{target}'
    path_model = f'{path}/model/{target}'
    labeled_csv_file = f'{data_path}/{des_type}.csv'
    model_file = f'{path_model}/xgb_{des_type}.pkl'
    # read
    df = pd.read_csv(labeled_csv_file, encoding='utf-8').dropna()
    test_df = df[df.iloc[:, -2] == 0]
    labels_lit = test_df.iloc[:, -1].values.astype(int)
    ifps_lit = test_df.iloc[:, 1:-2].values
    model_lit = xgb_model(model_file=model_file, ifps=ifps_lit, labels=labels_lit)

    # metric
    # strict on strict
    y_pred, y_pred_proba = model_strict.predict(ifps_strict)
    print('\n### performance of strict model on strict set###')
    model_strict.metric(y_pred_proba, y_pred, labels_strict)
    # strict on dude
    y_pred, y_pred_proba = model_strict.predict(ifps_dude)
    print('\n### performance of strict model on dude set###')
    model_strict.metric(y_pred_proba, y_pred, labels_dude)
    # strict on lit
    y_pred, y_pred_proba = model_strict.predict(ifps_lit)
    print('\n### performance of strict model on lit set###')
    model_strict.metric(y_pred_proba, y_pred, labels_lit)
    # dude on dude
    y_pred, y_pred_proba = model_dude.predict(ifps_dude)
    print('\n### performance of dude model on dude set###')
    model_dude.metric(y_pred_proba, y_pred, labels_dude)
    # dude on strict
    y_pred, y_pred_proba = model_dude.predict(ifps_strict)
    print('\n### performance of dude model on strict set###')
    model_dude.metric(y_pred_proba, y_pred, labels_strict)
    # dude on lit
    y_pred, y_pred_proba = model_dude.predict(ifps_lit)
    print('\n### performance of dude model on lit set###')
    model_dude.metric(y_pred_proba, y_pred, labels_lit)
    # lit on lit
    y_pred, y_pred_proba = model_lit.predict(ifps_lit)
    print('\n### performance of lit model on lit set###')
    model_lit.metric(pred_proba=y_pred_proba, pred_y=y_pred, y_true=labels_lit)
    # lit on strict
    y_pred, y_pred_proba = model_lit.predict(ifps_strict)
    print('\n### performance of lit model on strict set###')
    model_lit.metric(pred_proba=y_pred_proba, pred_y=y_pred, y_true=labels_strict)
    # lit on dude
    y_pred, y_pred_proba = model_lit.predict(ifps_dude)
    print('\n### performance of lit model on dude set###')
    model_lit.metric(pred_proba=y_pred_proba, pred_y=y_pred, y_true=labels_dude)


def aldh1_model():
    # init
    path = r'/home/xujun/Project_5'
    target = 'aldh1'
    des_type = 'ecfp_sifp'  # 'ecfp'  SP plec smina_nn
    data_path = f'{path}/cal_descriptors/v_1/{target}'
    path_model = f'{path}/model/{target}'
    # file
    labeled_csv_file = f'{data_path}/{des_type}.csv'
    model_file = f'{path_model}/xgb_{des_type}_py36.pkl'
    # read
    df = pd.read_csv(labeled_csv_file, encoding='utf-8').dropna()
    # preprocess
    if des_type in ['SP', 'SP_ifp', 'smina_nn']:
        print('data preprocess')
        # preprocess
        scaler = MinMaxScaler()
        des = df.iloc[:, 1:-2].values
        des = scaler.fit_transform(des)
        df_des = pd.DataFrame(des, columns=df.iloc[:, 1:-2].columns)
        df = pd.concat(
            [pd.DataFrame(df.iloc[:, 0].reset_index(drop=True)), df_des, df.iloc[:, -2:].reset_index(drop=True)],
            axis=1)
    # split
    train_df = df[df.iloc[:, -2] == 1]
    test_df = df[df.iloc[:, -2] == 0]
    # training
    labels = train_df.iloc[:, -1].values.astype(int)
    ifps = train_df.iloc[:, 1:-2].values
    xgb = xgb_model(model_file=model_file, ifps=ifps, labels=labels)
    # validation
    labels = test_df.iloc[:, -1].values.astype(int)
    ifps = test_df.iloc[:, 1:-2].values
    y_pred, y_pred_proba = xgb.predict(ifps)
    # metric
    xgb.metric(pred_proba=y_pred_proba, pred_y=y_pred, y_true=labels)

def aldh1_validation():
    # init
    path = r'/home/xujun/Project_5'
    target = 'aldh1'
    des_types = ['smina_nn', 'SP', 'SP_ifp', 'plec', 'ecfp', 'ecfp_sifp']
    for des_type in des_types:
        # des_type = 'smina_nn'  # 'ecfp'  SP plec smina_nn
        data_path = f'{path}/cal_descriptors/v_1/{target}'
        path_model = f'{path}/model/{des_type}'
        # file
        labeled_csv_file = f'{data_path}/{des_type}.csv'
        model_file = f'{path_model}/xgb_{des_type}.pkl'
        # model
        xgb = xgb_model(model_file=model_file, ifps='', labels='')
        # read
        df = pd.read_csv(labeled_csv_file, encoding='utf-8').dropna()
        # preprocess
        if des_type in ['SP', 'SP_ifp', 'smina_nn']:
            print('data preprocess')
            # preprocess
            scaler = MinMaxScaler()
            des = df.iloc[:, 1:-2].values
            des = scaler.fit_transform(des)
            df_des = pd.DataFrame(des, columns=df.iloc[:, 1:-2].columns)
            df = pd.concat(
                [pd.DataFrame(df.iloc[:, 0].reset_index(drop=True)), df_des, df.iloc[:, -2:].reset_index(drop=True)],
                axis=1)
        # split
        test_df = df[df.iloc[:, -2] == 0]
        # validation
        labels = test_df.iloc[:, -1].values.astype(int)
        ifps = test_df.iloc[:, 1:-2].values
        y_pred, y_pred_proba = xgb.predict(ifps)
        # metric
        print(f'strict on lit: {des_type}')
        xgb.metric(pred_proba=y_pred_proba, pred_y=y_pred, y_true=labels)

        # aldh1
        model_file = f'{path}/model/{target}/xgb_{des_type}.pkl'
        labeled_csv_file = f'{path}/model/{des_type}/{des_type}_top0.csv'
        # model
        xgb = xgb_model(model_file=model_file, ifps='', labels='')
        # read
        df = pd.read_csv(labeled_csv_file, encoding='utf-8').dropna()
        # preprocess
        if des_type in ['SP', 'SP_ifp', 'smina_nn']:
            print('data preprocess')
            # preprocess
            scaler = MinMaxScaler()
            des = df.iloc[:, 1:-2].values
            des = scaler.fit_transform(des)
            df_des = pd.DataFrame(des, columns=df.iloc[:, 1:-2].columns)
            df = pd.concat(
                [pd.DataFrame(df.iloc[:, 0].reset_index(drop=True)), df_des, df.iloc[:, -2:].reset_index(drop=True)],
                axis=1)
        # split
        _, test_df = train_test_split(df, train_size=0.8, shuffle=True,
                                             stratify=df.iloc[:, -1].values,
                                             random_state=42)
        # validation
        labels = test_df.iloc[:, -1].values.astype(int)
        ifps = test_df.iloc[:, 1:-2].values
        y_pred, y_pred_proba = xgb.predict(ifps)
        # metric
        print(f'lit on strict: {des_type}')
        xgb.metric(pred_proba=y_pred_proba, pred_y=y_pred, y_true=labels)

if __name__ == '__main__':
    train_a_xgb()
