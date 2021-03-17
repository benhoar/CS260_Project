import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import random, time
import csv

class StockFeaturesDataset(td.Dataset):
    def __init__(self, csv_file, binary):
        self.stocks_frame = []
        with open(csv_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                for idx, item in enumerate(row):
                    if item == 'Strong Buy' and binary:
                        row[idx] = '2.0'
                    elif item == 'Strong Buy' and not binary:
                        row[idx] = '1.0'
                    elif item == 'True' or item == 'Buy' or item == 'Strong Buy':
                        row[idx] = '1.0'
                    elif item == 'False' or item == 'Pass':
                        row[idx] = '0.0'
                row = [float(x) for x in row[1:]]
                self.stocks_frame.append(row)

    def __len__(self):
        return len(self.stocks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        indicators = self.stocks_frame[idx][:-1]
        indicators = np.array(indicators)

        label = self.stocks_frame[idx][-1]
        label = np.array(label)
        
        sample = (indicators, label)
        return sample

def stock_loaders(batch_size, file_path, binary, shuffle_test=False): 
    stock_dataset = StockFeaturesDataset(file_path, binary)

    train_num = int(np.floor(len(stock_dataset) * 0.8))

    train, test = td.random_split(stock_dataset, [train_num, len(stock_dataset) - train_num])

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
        shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader