import numpy as np
import pandas as pd
import os

class GetDataSet(object):
    def __init__(self, dataSetName):
        self.name = dataSetName
        if self.name=='tf':
            self.getTf()
        else:
            if self.name=='region':
                self.name='sheet_name'
            self.getData()
    def getTf(self):
        f_path='/home/u2022/nfs/waq/虫草/data/all.csv'
        data = pd.read_csv(f_path)
        # 查看哪些行存在 NaN 值
        nan_rows = np.nonzero(data.isna().any(axis=1))[0]
        # 删除
        data = data.drop(nan_rows, axis=0)
        df=data
        # 选择值为1的行的索引
        index_ones = df[df['label'] == 1].index

        # 从索引中随机选择500个
        random_ones = np.random.choice(index_ones, size=500, replace=False)

        # 根据随机选择的索引选择行
        selected_ones = df.loc[random_ones]

        # 选择值为0的行
        zeros = df[df['label'] == 0]

        # 将两个DataFrame拼接在一起
        result = pd.concat([selected_ones, zeros])
        result=result.reset_index(drop=True)
        return result
    def getData(self):
        f_path='/home/u2022/nfs/waq/虫草/data/true_label.csv'
        data = pd.read_csv(f_path)
        # 查看哪些行存在 NaN 值
        # nan_rows = data[data.isna().any(axis=1)]
        result = data.drop([1434, 3238,114,115,116,117], axis=0)

        labels = result[self.name].unique().tolist()

        # 创建标签到数字编码的映射
        label2idx = {label: idx for idx, label in enumerate(labels)}

        # 将标签转换为数字编码
        new_label='label_'+self.name
        result[new_label] = result[self.name].map(label2idx)
        # result['label_region'] = result["sheet_name"].apply(lambda x: label2idx[x])

        print(result[new_label].unique())
        # print(result)
        result=result.reset_index(drop=True)
        return result