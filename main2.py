import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import imageio
import torch.utils.data as Data

import datetime
import matplotlib.pyplot as plt
import add_noise as an
import create_noises as cn

#vasatie tedade chanel hast k baraye ma aval yeke
#bad b tedad node haye hidden mishe

print("main2")
data = np.random.normal(0, 1, (2, 50))
data = data.astype("float32")
data = torch.from_numpy(data)
data = data.reshape((2, 1, 50))
y = np.random.normal(0, 1, 2)
y = y.astype("float32")
y = torch.from_numpy(y)
y = y.reshape((1, 1, 2))


class Net(nn.Module):
    def __init__(self, num_hidden, num_input=1, batch_size=1):
        super(Net, self).__init__()
        self.hiddenLayer = nn.Conv1d(in_channels=num_input, out_channels=num_hidden, kernel_size=50)
        self.outPut = nn.Linear(num_hidden, 1)

    def forward(self, x):
        print(x.size())
        x = self.hiddenLayer(x)
        print(x.size())
        x = F.relu(x)
        print(x.size())
        print("here")
        x = x.view(x.size(0), -1)
        x = self.outPut(x)
        print(x)

        return x


t_net = Net(num_hidden=10, num_input=1, batch_size=2)


optimizer = torch.optim.SGD(t_net.parameters(), lr=0.2)
loss_func = nn.MSELoss()

print("meeeeeeeee")
for name, param in t_net.state_dict().items():
    if name == "outPut.weight":
        print(param.size())


# train the network
for t in range(1):
    prediction = t_net(data)  # input x and predict based on x
    print(prediction.size())
    #print(y)
    loss = loss_func(prediction, y)  # must be (1. nn output, 2. target)

    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients
