import torch
import create_noises as cn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import add_noise as ad
import my_sgd as MySGD


f = __import__("load data")
a = torch.load("C:/Users/USER/PycharmProjects/pythonProject/towards-the-importance-of-noise-in-nueral-networks/weights.pth")


class StudentNet(nn.Module):
    def __init__(self, num_hidden, num_input=1, kernel=1, k=1):
        super(StudentNet, self).__init__()
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


def initialization(model, weight_range):
    state_dict = model.state_dict()
    for name, param in state_dict.items():

        # print("before")
        # print(name)
        # print(param)

        if name == "outPut2.weight":
            tt = np.random.uniform(0, weight_range, param.shape)
            tt = torch.from_numpy((tt))
            state_dict[name].copy_(tt)
        elif name == "hiddenLayer.weight":
            tt = np.random.uniform(0, 1, param.shape)
            tt = torch.from_numpy((tt))
            state_dict[name].copy_(tt)


k = 100
m = 2
d = k*m
print(a["outPut2.weight"])
ones = np.ones(a["outPut2.weight"].shape)
B0_initialization = np.abs(np.dot(a["outPut2.weight"], ones.T)/ k)
print(a['hiddenLayer.weight'].shape)
#print(np.dot(a["outPut2.weight"], ones.T)**2/ np.linalg.norm(a["outPut2.weight"]))
x = np.zeros((1, 100))
print("norm")
ones_x = np.ones(x.shape)
print(np.dot(a["outPut2.weight"], ones_x.T)**2/ (np.linalg.norm(a["outPut2.weight"]))**2)

s_net = StudentNet(num_hidden=k, num_input=1, kernel=m, k=k) #Creating teacher network

#
# optimizer = MySGD.MySGD(s_net.parameters(), lr=0.2)
# loss_func = nn.MSELoss()
# #
# #
# data_train_file = f.GenerateData("data.csv", "y.csv")
# data_train = f.GeneratedDataLoader(data_train_file, bs=4)
# #
# for epoch in range(2):
#    print(epoch)
#    initialization(s_net, weight_range=B0_initialization)
#     for i in range(125):
#         # [big_ep, small_ep] = cn.random_ep(p_a=, p_w=,)
#         inp = data_train.next_batch()
#         prediction = s_net(inp["data"])  # input x and predict based on x
#         loss = loss_func(prediction, inp["y"])  # must be (1. nn output, 2. target)
#         optimizer.zero_grad()  # clear gradients for next train
#         # ad.add_noise(s_net, big_ep, small_ep)
#         loss.backward()  # backpropagation, compute gradients
#         optimizer.step()  # apply gradients
# #         # if (i + 1) % 5 == 0:
# #             # board.add_scalar("metric" + str(epoch + 1), loss.item(), i + 1)
# #
#
