import torch
import torch.nn as nn
from model import MLP_core
from dataset import train_loader, val_loader, test_loader
import matplotlib.pyplot as plt
from pathlib import Path
#实例化模型
model = MLP_core(input_dim=6)   
#定义损失函数#MSE= 类似于我们的线性回归里面的loss函数，其实就是一样的
criterion = nn.MSELoss()
#定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#本质上还是线性回归Loss梯度下降那一块
loss_history = []
val_loss_history = []

for epoch in range(100):
    model.train()
    running_loss = 0.0
    

    for X_batch, y_real in train_loader:#32批次
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_real)
        optimizer.zero_grad()#清空上一次的梯度
        loss.backward()#计算着一次的梯度
        optimizer.step()#更新梯度

        running_loss += loss.item()*X_batch.size(0)#神经网络loss的综合
    average_loss=running_loss / len(train_loader.dataset)
    loss_history.append(average_loss)
    print(f"Epoch {epoch + 1} / 1000, loss = {average_loss:.3f}")
    average_loss=0



#验证区
val_running_loss=0
loss_val=[]
model.eval()
with torch.no_grad():
    val_running_loss = 0.0
    for X_batch ,y_real in val_loader:
         y_pred=model(X_batch)
         loss=criterion(y_pred,y_real)
         error=y_pred-y_real
         val_running_loss += loss.item()*X_batch.size(0)#神经网络loss的综合
    val_average_loss=val_running_loss / len(val_loader.dataset)
    print(f"验证的loss = {val_average_loss:.3f}")
    loss_val.append(val_average_loss)
    
test_running_loss=0

model.eval()
with torch.no_grad():
    for X_batch ,y_real in test_loader:
        y_pred=model(X_batch)
        loss=criterion(y_pred,y_real)
        test_running_loss += loss.item()*X_batch.size(0)#神经网络loss的综合
    test_average_loss=test_running_loss / len(test_loader.dataset)
    print(f'本次测试的Loss是:{test_average_loss}')


plt.figure()
plt.plot(loss_history, label="train", color="#1f1e33")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("MLP")
plt.legend()
output_dir = Path(__file__).with_name("output")
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "loss.png"
plt.savefig(output_file, dpi=1200, bbox_inches="tight")
plt.close()
