import torch
from torch.utils.data.dataset import Dataset

import pandas as pd

class MyDataset(Dataset):
    def __init__(self, path):
        label_col = [
            'Input_A6_024' ,'Input_A3_016', 'Input_C_013', 'Input_A2_016', 'Input_A3_017',
            'Input_C_050', 'Input_A6_001', 'Input_C_096', 'Input_A3_018', 'Input_A6_019',
            'Input_A1_020', 'Input_A6_011', 'Input_A3_015', 'Input_C_046', 'Input_C_049',
            'Input_A2_024', 'Input_C_058', 'Input_C_057', 'Input_A3_013', 'Input_A2_017'
        ]
        df = pd.read_csv(path)
        df = df.fillna(0)
        self.data = df.drop(label_col, axis=1)
        self.data2 = df.drop(label_col, axis=1)

        #  Mean normalization [-1, 1]
        for i in self.data.columns:
            _max = self.data[i].max()
            _min = self.data[i].min()
            mu =self.data[i].mean()
            if _max - _min != 0:
                self.data[i] = (self.data[i] - mu) / (_max - _min)
            
        self.target = df[label_col]


    def __getitem__(self, index):
        return self.data.iloc[index].values, self.target.iloc[index].values
    
    def __len__(self):
        return len(self.data)