#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2021/4/12 19:05
# @author : Xujun Zhang


import pandas as pd


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



