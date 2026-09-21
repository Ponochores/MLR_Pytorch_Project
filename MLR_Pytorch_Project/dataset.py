import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import time
1.#预处理
t1=time.time()
df=pd.read_csv("弹幕威力数据.csv")
df.info()
2.#预处理：结果和输入
#columns代表的是列名，而drop的含义是删除
# values负责把Series转成NumPy数组
X = df.drop(columns=["power"])
Y = df["power"].values.reshape(-1, 1)
X.info()

#独热编码
X_color = pd.get_dummies(X[["color"]],columns=["color"])#数据来源和需要处理的数据

###
X_color.info()
X_other=X.drop(columns=["color"])
#print(X_color)
#print(X_other)
# 对数值特征做标准化（一定要做，不错会挂）
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_other),
    columns=X_other.columns, #保持原来的列名不变
    index=X_other.index #保持原来的索引，拒绝加入01
)
#print(X_scaled)
#终于处理好了2026.9.16.21:41
X_final=pd.concat([X_scaled,X_color],axis=1)#axis是轴的含义规定拼接方向的
print(X_final)
#2026.9.16 21:56
3.#划分数据集(明天来做)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_final,
    Y,
    train_size=0.6,
    random_state=42,#相当于随机种子
    shuffle=True#随机打乱
)

X_test,X_val, y_test,y_val= train_test_split (
    X_temp,
    y_temp,
    train_size=0.5,
    random_state=42,#这里的写法挺像C语言的
    shuffle=True
)

4.# 转换为 PyTorch Tensor
# Tensor 是 PyTorch 模型能够直接接收的数据格式。而且直接收.32浮点数
X_train_tensor = torch.tensor(X_train.to_numpy(dtype=np.float32), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val.to_numpy(dtype=np.float32), dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.to_numpy(dtype=np.float32), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

5.# 封装数据集，并按 batch 提供给训练过程。相当于把订书机把xy钉在一起了。防止后续训练的时候不匹配。
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
#用于批量测试
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#检验一下化为张量的集合
print("训练集：", X_train_tensor.shape, y_train_tensor.shape)
print("验证集：", X_val_tensor.shape, y_val_tensor.shape)
print("测试集：", X_test_tensor.shape, y_test_tensor.shape)
t2=time.time()
print(round(t2-t1,3))#检验一下程序运行的时间，结果发现，其实运行速度真的挺慢的，本地要将近40ms才能运行完（我的CPU是ARM M5）
print(train_loader)#输出的应当是内存空间