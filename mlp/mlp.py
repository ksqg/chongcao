import numpy as np
from numpy import vstack
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.utils.data as Data
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from torch.nn import BCELoss
# audio_features = pd.DataFrame(np.squeeze(np.load( '/home/u2022/nfs/waq/audio_regression/Features/whole_samples_reg_op_03.npz')['arr_0']))
f_path='/home/u2022/nfs/waq/虫草/data/all.csv'
data = pd.read_csv(f_path)
# 查看哪些行存在 NaN 值
nan_rows = data[data.isna().any(axis=1)]

# 打印存在 NaN 值的行
# print(nan_rows)
# 删除
data = data.drop([1434, 3238], axis=0)
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

# 输出结果
print(result)
result=result.reset_index(drop=True)
# 标准化
def SS(data):
    """
        :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after StandScaler :(n_samples, n_features)
       """
    data=data.values
    scale = StandardScaler()
    df = scale.fit_transform(data)
    df=pd.DataFrame(df)
    return df

from scipy.signal import savgol_filter
def SG(data):
    cols = data.columns[2:3602]
    rows = range(0, data.shape[0])
    # 对每一行的特定列进行Savitzky-Golay平滑处理
    for row in rows:
        data.loc[row, cols] = savgol_filter(data.loc[row, cols], window_length=11, polyorder=2)
    return data
from sklearn.cross_decomposition import PLSRegression

# 假设你有一个名为df的DataFrame，其中包含需要进行MSC的光谱数据列
pls = PLSRegression(n_components=10)

# 定义一个函数，用于将每一行的光谱数据进行MSC
def MSC(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after MSC :(n_samples, n_features)
    """
    data=data.values
    n, p = data.shape
    msc = np.ones((n, p))

    for j in range(n):
        mean = np.mean(data, axis=0)

    # 线性拟合
    for i in range(n):
        y = data[i, :]
        l = LinearRegression()
        l.fit(mean.reshape(-1, 1), y.reshape(-1, 1))
        k = l.coef_
        b = l.intercept_
        msc[i, :] = (y - b) / k
    msc=pd.DataFrame(msc) 
    return msc



# result=SG(result)
# result=MSC(result)
features=result.loc[:,'2':'3602']
# features=SS(features)
targets = pd.DataFrame(result['label'])

# print(features,targets)
print(features)
from sklearn import preprocessing  
 
# prepare the dataset
def prepare_data():
    # min_max_scaler = preprocessing.MinMaxScaler()  
    
    # Y = min_max_scaler.fit_transform(targets.values.reshape(-1, 1))  
    # Y=pd.DataFrame(Y)
    # 切分训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(audio_features.iloc[:,0:256],audio_targets.iloc[:,1].astype('float32'),
    #                                                     test_size = 0.2, random_state = 42)
    X_train, X_test, y_train, y_test = train_test_split(features,targets.iloc[:,0].astype('float32'),
                                                        test_size = 0.2, random_state = 42)

    # # # 数据标准化处理
    # # scale = StandardScaler()
    # # X_train_s = scale.fit_transform(X_train)
    # # X_test_s = scale.transform(X_test)

    # # print(X_test_s)
    # # 将数据集转为张量
    # X_train_t = torch.from_numpy(X_train.astype(np.float32).values)
    # y_train_t = torch.from_numpy(y_train.astype(np.float32).values)
    # X_test_t = torch.from_numpy(X_test.astype(np.float32).values)
    # y_test_t = torch.from_numpy(y_test.astype(np.float32).values)
    
    scale = StandardScaler()
    X_train_s = scale.fit_transform(X_train)
    X_test_s = scale.transform(X_test)   

    # print(X_test_s)
    # 将数据集转为张量
    X_train_t = torch.from_numpy(X_train_s.astype(np.float32))
    y_train_t = torch.from_numpy(y_train.astype(np.float32).values)
    X_test_t = torch.from_numpy(X_test_s.astype(np.float32))
    y_test_t = torch.from_numpy(y_test.astype(np.float32).values)
    
    
    # 将训练数据处理为数据加载器
    train_data = Data.TensorDataset(X_train_t, y_train_t)
    test_data = Data.TensorDataset(X_test_t, y_test_t)
    # prepare data loaders
    train_loader = Data.DataLoader(dataset = train_data, batch_size = 64, 
                                shuffle = True, num_workers = 1)
    test_loader = Data.DataLoader(dataset = test_data, batch_size = 1024, 
                                shuffle = True)
    
    return train_loader, test_loader
# 搭建全连接神经网络回归
class MLPregression(nn.Module):
    def __init__(self):
        super(MLPregression, self).__init__()
        # 第一个隐含层
        self.hidden1 = nn.Linear(in_features=3601, out_features=64, bias=True)
        # 第二个隐含层
        self.hidden2 = nn.Linear(64, 64)
        # 第三个隐含层
        self.hidden3 = nn.Linear(64, 1)
        # 第4个隐含层
        # self.hidden4 = nn.Linear(32, 1)
        # 回归预测层
        self.predict = nn.Sigmoid()
        
    # 定义网络前向传播路径
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        # x = F.relu(self.hidden4(x))
        output = self.predict(x)
        # 输出一个一维向量
       
        return output[:, 0]


# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = BCELoss()
    # 定义优化器
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    # enumerate epochs
    for epoch in range(100):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            print("epoch: {}, batch: {}, loss: {}".format(epoch, i, loss.data))
            # update model weights
            optimizer.step()
   


# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = [], []
    correct = 0
    total = 0
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        # yhat = yhat.detach().numpy()
        # actual = targets.numpy()
        # # actual = actual.reshape((len(actual), 1))
        # # round to class values
        # yhat = yhat.round()
        # # store
        
        # predictions.append(yhat)
        # actuals.append(actual)
        predictions = torch.round(yhat)
        total += targets.size(0)
        correct += (predictions == targets).sum().item()
    accuracy = correct / total
    # print('Accuracy: {:.2f}%'.format(100 * accuracy))
    # predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    # print(actuals, predictions)
    # acc = accuracy_score(actuals, predictions)
    return accuracy


model = MLPregression()
train_dl, test_dl = prepare_data()
# print(model)
# train the model
train_model(train_dl, model)
# evaluate the model
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)
