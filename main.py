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

data = np.random.normal(0, 1, 100)

#Generating a data for training
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)


#Showing the generated Data
plt.plot(x, y, color='orange', marker='o', linestyle='dashed')
plt.show()

# Design a simple neural network with just a hidden layer
class simple_net(nn.Module):
    def __init__(self, num_hidden, num_input=1):
        super(simple_net, self).__init__()

        self.hiddenLayer = nn.Linear(num_input, num_hidden, bias=False)
        self.outPut = nn.Linear(num_hidden, 1, bias=False)

    def forward(self, x):
        x = F.relu(self.hiddenLayer(x))
        x = self.outPut(x)

        return x


# Create the network to train
t_net = simple_net(num_hidden=10, num_input=1)
#initialization(t_net)

#a_star = 10
#k = 20
#big_epsilon, small_epsilon = cn.random_ep(p_a=3, p_w=1, a_size=output_layer_size, weights_size=hidden_layer_size)
#print("big")
#print(big_epsilon)
#print("small")
#print(small_epsilon)


optimizer = torch.optim.SGD(t_net.parameters(), lr=0.2)
loss_func = nn.MSELoss()

my_images = []
fig, ax = plt.subplots(figsize=(12, 7))

# train the network
for t in range(1):
    prediction = t_net(x)  # input x and predict based on x

    loss = loss_func(prediction, y)  # must be (1. nn output, 2. target)

    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

