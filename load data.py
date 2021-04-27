from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import numpy as np
import pandas as pd


class GenerateData(Dataset):
    #im shows the dimension of the generated data
    def __init__(self, data_csv_file, y_csv_file):
        self.data = pd.read_csv(data_csv_file, dtype=float)
        self.data = self.data.drop(self.data.columns[0], axis=1)
        self.data = self.data.to_numpy()
        self.y = pd.read_csv(y_csv_file, dtype=float)
        self.y = self.y.drop(self.y.columns[0], axis=1)
        self.y = self.y.to_numpy()

    def __getitem__(self, index):
        d = self.data[index]
        d = d.astype("float32")
        d = torch.from_numpy(d)
        d = d.unsqueeze(0)

        y = self.y[index]
        y = y.astype("float32")
        y = torch.from_numpy(y)

        sample = {"data": d, "y": y}
        return sample
        # data = self.data.astype("float32")
        # data = torch.from_numpy(data)
        # data = data.reshape((self.batch_size, 1, self.data_size))
        #
        # y = self.y.astype("float32")
        # y = torch.from_numpy(y)
        # y = y.reshape((self.batch_size, 1))
        #
        # sample = {"data": data, "y": y}

        #return sample

    def __len__(self):
        return len(self.data)


dd = GenerateData("data.csv", "y.csv")
# ss = dd.__getitem__(0)
# print(ss[0]
#       )


class GeneratedDataLoader(object):
    def __init__(self, dataset, bs):
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


bb = GeneratedDataLoader(dd, bs=4)
print(dd.__getitem__(0)["y"])
f_batch = bb.next_batch()
print("here")
batch = f_batch["data"]
print(batch.size())

# item = dd.__getitem__(0)
# print(item)

#generate Data
# data = np.random.normal(0, 1, (500, 50))
# x_df = pd.DataFrame(data)
# x_df.to_csv('data.csv', float_format='%.3f')
# y = np.random.normal(0, 1, 500)
# x_df = pd.DataFrame(data)
# x_df.to_csv('y.csv', float_format='%.3f')

#
# for t in range(1):
#    inp = bb.next_batch()
#    print(len(inp))