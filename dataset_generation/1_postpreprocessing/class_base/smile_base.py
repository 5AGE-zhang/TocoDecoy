#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2021/4/12 15:47
# @author : Xujun Zhang

import csv
import os
import pandas as pd
# import h5py
import numpy as np
import oddt
from oddt import fingerprints
from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger
from rdkit.Chem import AllChem, Descriptors, Fragments
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
RDLogger.DisableLog('rdApp.*')

