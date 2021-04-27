import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import MyDataset
from torch.utils.data import DataLoader
from model import MyModel

import os
import math
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils import WRMSE

# GPU = 0
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
batch_size = 64
epoches = 30
learning_rate = 0.01
# input_size = 223
# output_size = 20
# dropout = 0.2

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, "train.csv")

# torch.cuda.set_device(GPU)

# set dataset and dataloader
dataset = MyDataset(data_path)

train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)])

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# set model
model = MyModel()
model = model.to(device)
# model = model.cuda()

# set optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

train_loss_list = list()
train_wrmse_list = list()
valid_loss_list = list()
valid_wrmse_list = list()
best = 100
for epoch in range(epoches):
    print(f'\nEpoch: {epoch+1}/{epoches}')
    print('-' * len(f'Epoch: {epoch+1}/{epoches}'))
    train_loss = 0.0
    train_wrmse = 0.0
    valid_loss = 0.0
    valid_wrmse = 0.0

    model.train()
    for data, target in train_dataloader:
        data, target = data.float().to(device), target.float().to(device)
        
        # forward + backward + optimize
        preds = model(data)
        loss = criterion(preds, target)
        wrmse = WRMSE(preds, target, device)
        # zero the parameter gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # print(loss.item())
        train_wrmse += wrmse

    train_loss /= len(train_dataset)
    train_wrmse = math.sqrt(train_wrmse/len(train_dataset))
    print(f'Training loss: {train_loss:.4f}, Training wrmse: {train_wrmse:.4f}')
    train_loss_list.append(train_loss)
    train_wrmse_list.append(train_wrmse)

    model.eval()
    for data, target in valid_dataloader:
        data, target = data.float().to(device), target.float().to(device)

        preds = model(data)
        loss = criterion(preds, target)
        wrmse = WRMSE(preds, target, device)

        valid_loss += loss.item()
        valid_wrmse += wrmse

    valid_loss /= len(valid_dataset)
    valid_wrmse = math.sqrt(valid_wrmse/len(valid_dataset))
    print(f'Validation loss: {valid_loss:.4f}, Validation wrmse: {valid_wrmse:.4f}')
    valid_loss_list.append(valid_loss)
    valid_wrmse_list.append(valid_wrmse)

    if train_wrmse < best :
            best = train_wrmse
            torch.save(model.state_dict(), 'weight.pth')

print('\nFinished Training')

pd.DataFrame({
    "train-loss": train_loss_list,
    "valid-loss": valid_loss_list
}).plot()
plt.savefig("Loss_Curve")

pd.DataFrame({
    "train-wrmse": train_wrmse_list,
    "valid-wrmse": valid_wrmse_list
}).plot()
plt.savefig("Wrmse_Curve")