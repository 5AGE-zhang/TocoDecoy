#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2021/2/19 9:38
# @author : Xujun Zhang

import os
import time
import asyncio
from tqdm import tqdm
from functools import partial
from my_utils import Smiles
from multiprocessing import Pool, Process

async def write_file(content, dst_file):
    with open(dst_file, 'w') as f:
        if format_ == 'sdf':
            f.write(content + f'{file_label[format_]}\n')
        else:
            # format == 'mol2'
            f.write(f'{file_label[format_]}\n' + content)


async def split_(content, dst_path, format_):

    if content != '':
        lig_name = content.split('\n')[0].strip()  # 获取小分子名字
        lig_file = '{}/{}.{}'.format(dst_path, lig_name, format_)  # 定义输出分子路径

        # 递归检查重名在文件
        def same_file(lig_file, n=0):
            if os.path.exists(lig_file):
                n += 1
                lig_file = '{}/{}_{}.{}'.format(dst_path, lig_name, n, format_)
                return same_file(lig_file, n)
            else:
                return lig_file

        lig_file = same_file(lig_file)  # 检查是否重名
        # 输出文件
        await write_file(content, lig_file)


def asy_split(contents, dst_path, format_):
    tasks = []
    for content in tqdm(contents):
        # no threading
        tasks.append(asyncio.ensure_future(split_(content, dst_path, format_)))
        # smiler.split_(content=content, dst_path=dst_path, format=format)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait(tasks))

if __name__ == '__main__':
    # init
    path = '/home/xujun/Project_5/mix'
    # job_type = 'fpb'  # pcb  fpb ['pcb', 'fpb']
    for job_type in ['aldh1']:
        similar_path = f'{path}/{job_type}'
        src_path = f'{similar_path}/docked'
        # format_ = 'mol2'
        for format_ in ['mol2', 'sdf']:
            dst_path = f'{similar_path}/ligands_{format_}'
            # smile
            smiler = Smiles(smile_lis=[''], names=[])
            raw_file = f'{src_path}/SP_raw.{format_}'
            file_label = {
                'sdf': '$$$$',
                'mol2': '@<TRIPOS>MOLECULE'
            }
            # split
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)
                # read file
                # 读取数据，存到con中
                with open(raw_file, 'r') as f:
                    con = f.read()
                # 根据@<TRIPOS>MOLECULE分割字符串
                con = con.split(f'{file_label[format_]}\n')
                # multi process
                n_job = 27  # 考虑到余数
                sep = len(con)//n_job
                contents = [con[i*sep: (i+1)*sep] for i in range(0, n_job+1)]
                start_time = time.perf_counter()
                asy_split_ = partial(asy_split, dst_path=dst_path, format_=format_)
                for i in range(n_job+1):
                    p = Process(target=asy_split_, args=(contents[i],))
                    p.start()
                # pool = Pool(n_job+1)
                # pool.map(asy_split_, contents)
                # pool.close()
                # pool.join()
                # loop.close()
                print((time.perf_counter()-start_time)/60)



