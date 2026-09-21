

import torch
import numpy as np
import torch.nn as nn

class MLP_core(nn.Module):
   def __init__(self,input_dim):
       super().__init__()#是初始化父类，必须写
       #把几个层按顺序串起来
       self.net=nn.Sequential(
           nn.Linear(input_dim,32),
           nn.ReLU(),

           nn.Linear(32,32),
           nn.ReLU(),
           nn.Linear(32,1),
   )
#上面的函数适用于来进行网络链接，非常人性化的链接方式，完全的自然语言。

   def forward(self,x):
       return self.net(x)#必须要规定，不然谁知道是向前还是向后呢？