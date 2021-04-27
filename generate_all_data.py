from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import numpy as np
import pandas as pd

data = np.random.normal(0, 1, (500, 200))
x_df = pd.DataFrame(data)
x_df.to_csv('data.csv', float_format='%.3f')
y = np.random.normal(0, 1, 500)
for i in range(0, 500):
    y[i] = np.sum(data[i])
print(y)
y_df = pd.DataFrame(y)
y_df.to_csv('y.csv', float_format='%.3f')