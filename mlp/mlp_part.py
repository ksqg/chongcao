import numpy as np
from numpy import vstack
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim import SGD
import torch.utils.data as Data
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from torch.nn import BCELoss
# audio_features = pd.DataFrame(np.squeeze(np.load( '/home/u2022/nfs/waq/audio_regression/Features/whole_samples_reg_op_03.npz')['arr_0']))
f_path='/home/u2022/nfs/waq/虫草/data/true_label.csv'
data = pd.read_csv(f_path)
# 查看哪些行存在 NaN 值
nan_rows = data[data.isna().any(axis=1)]

# 打印存在 NaN 值的行
# print(nan_rows)
# 删除
print(data[data['label_part']=='b'])
result = data.drop([1434, 3238,114,115,116,117], axis=0)

print(result)
result=result.reset_index(drop=True)

# cols = result.columns[2:3602]
# rows = range(0, 667)

# # 对每一行的特定列进行Savitzky-Golay平滑处理
# for row in rows:
#     result.loc[row, cols] = savgol_filter(result.loc[row, cols], window_length=11, polyorder=2)
    
features=result.loc[:,'2':'3602']
targets = pd.DataFrame(result['label_part'])

# print(features,targets)
from sklearn import preprocessing  
 
# prepare the dataset
def prepare_data():
    min_max_scaler = preprocessing.MinMaxScaler()  
    
      
    
    # 切分训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(audio_features.iloc[:,0:256],audio_targets.iloc[:,1].astype('float32'),
    #                                                     test_size = 0.2, random_state = 42)
    X_train, X_test, y_train, y_test = train_test_split(features,targets.iloc[:,0],
                                                        test_size = 0.2, random_state = 42)
    # 数据标准化处理
    # 数据标准化处理

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

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out

# 定义训练函数
def train_model(model, criterion, optimizer, train_loader, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.float()
            labels = labels - 1
            labels = labels.long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, running_loss / len(train_loader)))

# 定义测试函数
def test_model(model, test_loader):
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs = inputs.float()
            labels = labels - 1
            labels = labels.long()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy: %.2f %%' % (100 * correct / total))

# 定义训练集和测试集
train_loader, test_loader = prepare_data()
# 定义模型参数
input_dim = 3601
hidden_dim = 64
output_dim = 5
num_epochs = 50
learning_rate = 0.01

# 初始化模型、损失函数和优化器
model = MLP(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练并测试模型
train_model(model, criterion, optimizer, train_loader, num_epochs)
test_model(model, test_loader)
# print('Accuracy: %.3f' % acc)
