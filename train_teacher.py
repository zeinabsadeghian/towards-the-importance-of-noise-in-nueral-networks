import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
f = __import__("load data")

import torchvision
import imageio

import datetime
import matplotlib.pyplot as plt
import add_noise as an
import create_noises as cn

#vasatie tedade chanel hast k baraye ma aval yeke
#bad b tedad node haye hidden mishe
#k = 10 #number of patches that we want to generate
m = 2 #m is the dimension of filter == p
#d = 50 # d is the dimension of input
k = 100
d = k*m



class Net(nn.Module):
    def __init__(self, num_hidden, num_input=1, kernel=1, k=1):
        super(Net, self).__init__()
        self.hiddenLayer = nn.Conv1d(in_channels=num_input, out_channels=num_hidden, kernel_size=kernel, stride=kernel)
        self.outPut = nn.AvgPool1d(kernel_size=k)
        self.outPut2 = nn.Linear(num_hidden, 1)

    def forward(self, x):
        # print(x.size())
        x = self.hiddenLayer(x)
        # print(x.size())
        x = F.relu(x)
        #print(x)
        # print("here")
        x = self.outPut(x)
        #print(x)
        x = x.view(x.size(0), -1)
        x = self.outPut2(x)
        #print(x)

        return x


t_net = Net(num_hidden=k, num_input=1, kernel=m, k=k) #Creating teacher network


optimizer = torch.optim.SGD(t_net.parameters(), lr=0.2)
loss_func = nn.MSELoss()
# board = SummaryWriter("C:/Users/USER/PycharmProjects/pythonProject/towards-the-importance-of-noise-in-nueral-networks/board")

x = np.zeros((1, 100))
for i in range(0, 99):
    x[0][i] = 0.2
# o = 0
# print("meeeeeeeee")
state_dict = t_net.state_dict()
for name, param in state_dict.items():
    print(name)
o = 0
for param in t_net.parameters():
    if o == 2:
        param.requires_grad = False
        param.copy_(torch.from_numpy(x))
    o = o + 1


board = SummaryWriter(
    "C:/Users/USER/PycharmProjects/pythonProject/towards-the-importance-of-noise-in-nueral-networks/board")
dd = f.GenerateData("data.csv", "y.csv")
bb = f.GeneratedDataLoader(dd, bs=4)
# train the network
for epoch in range(500):
    #print(epoch)
    for i in range(125):
        inp = bb.next_batch()
        # print(len(inp["data"]))
        prediction = t_net(inp["data"])  # input x and predict based on x
        # print(prediction.size())
        loss = loss_func(prediction, inp["y"])  # must be (1. nn output, 2. target)

        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        if (epoch + 1) % 5 == 0:
            board.add_scalar("metric", loss.item(), epoch + 1)

#python -m tensorboard.main --logdir=C:\Users\USER\PycharmProjects\pythonProject\towards-the-importance-of-noise-in-nueral-networks\board

torch.save(t_net.state_dict(), "C:/Users/USER/PycharmProjects/pythonProject/towards-the-importance-of-noise-in-nueral-networks/weights.pth")

for param in t_net.parameters():
    print(param)
