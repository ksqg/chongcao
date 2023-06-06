import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.datasets import fetch_california_housing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.utils.data as Data
import matplotlib.pyplot as plt

# audio_features = pd.DataFrame(np.squeeze(np.load( '/home/u2022/nfs/waq/audio_regression/Features/whole_samples_reg_op_03.npz')['arr_0']))
f_path='/home/u2022/nfs/waq/虫草/data/true.csv'
features = pd.read_csv(f_path)
features['label']=1
f2_path='/home/u2022/nfs/waq/虫草/data/false.csv'
features2 = pd.read_csv(f2_path)
features2['label']=0
df=pd.concat([features,features2],axis=0)
df = df.reset_index(drop=True)
print(df)
df.to_csv('./data/all.csv', index=False,encoding='utf-8-sig')