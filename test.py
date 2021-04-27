import torch
from model import MyModel
from dataset import MyDataset

import os
import pandas as pd

base_path = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(base_path, "test.csv")

train_data_path = os.path.join(base_path, "train.csv")
train_dataset = MyDataset(train_data_path)

# load model and use weights we saved before.
model = MyModel()
model.load_state_dict(torch.load('weight.pth', map_location='cpu'))
model.eval()
# load testing data
data = pd.read_csv(test_data_path, encoding='utf-8')
label_col = [
    'Input_A6_024' ,'Input_A3_016', 'Input_C_013', 'Input_A2_016', 'Input_A3_017',
    'Input_C_050', 'Input_A6_001', 'Input_C_096', 'Input_A3_018', 'Input_A6_019',
    'Input_A1_020', 'Input_A6_011', 'Input_A3_015', 'Input_C_046', 'Input_C_049',
    'Input_A2_024', 'Input_C_058', 'Input_C_057', 'Input_A3_013', 'Input_A2_017'
]
# ================================================================ #
# if do some operations with training data,
# do the same operations to the testing data in this block
data = data.fillna(0)
for i in data.columns:
    _max = train_dataset.data2[i].max()
    _min = train_dataset.data2[i].min()
    mu =train_dataset.data2[i].mean()
    if _max - _min != 0:
        data[i] = (data[i] - mu) / (_max - _min)
# ================================================================ #
# convert dataframe to tensor, no need to rewrite
inputs = data.values
inputs = torch.tensor(inputs)
# predict and save the result
result = pd.DataFrame(columns=label_col)
outputs = model(inputs.float())
for i in range(len(outputs)):
    tmp = outputs[i].detach().numpy()
    tmp = pd.DataFrame([tmp], columns=label_col)
    result= pd.concat([result, tmp], ignore_index=True)
result.to_csv('result.csv', index=False)
