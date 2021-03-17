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
    def __init__(self, csv_file):
        self.stocks_frame = []
        with open(csv_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                for idx, item in enumerate(row):
                    if item == 'True' or item == 'Buy' or item == 'Strong Buy':
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

def stock_loaders(batch_size, shuffle_test=False): 
    stock_dataset = StockFeaturesDataset('Stocks_forML_Feb24.csv')

    train_num = int(np.floor(len(stock_dataset) * 0.8))

    train, test = td.random_split(stock_dataset, [train_num, len(stock_dataset) - train_num])

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
        shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader

batch_size = 64
test_batch_size = 64

train_loader, _ = stock_loaders(batch_size)
_, test_loader = stock_loaders(test_batch_size)

# The number of epochs is at least 10, you can increase it to achieve better performance
num_epochs = 10
learning_rate = 0.01

print("Starting training for Linear SVM")
# Define the model for fully-connected with ReLU
class SVM(nn.Module):
    def __init__(self):
        super(SVM, self).__init__()
        self.svm = nn.Linear(26, 1)

    def forward(self, x):
        return self.svm(x)   

model = SVM()

# For training on GPU
# model = model.cuda()

def my_hinge_loss(output, target):
    loss = 1 - torch.multiply(output,target)
    loss = torch.clamp(loss,min=0.0,max=float('inf'))
    return torch.mean(loss)


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.HingeEmbeddingLoss()
epoch_num = 1

# Training the Model
for epoch in range(num_epochs):
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        # For training on GPU
        # images = images.cuda()
        # labels = labels.cuda()
        labels = (2*(labels.float()-0.5))
        # Compute predicted value
        y_pred = model(images.float()).squeeze(1)
        
        # Compute loss
        loss = criterion(y_pred, labels)
        total_loss += loss.item()

        # Update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ## Print your results every epoch
    print("Epoch {} Avg Loss: {}".format(epoch_num, total_loss/len(train_loader)))
    epoch_num += 1

# Test the Model
correct = 0.
total = 0.
for images, labels in test_loader:
    # For training on GPU
    # images = images.cuda()
    # labels = labels.cuda()
    outputs = model(images.float()).squeeze(1)
    # predicted = torch.round(torch.sigmoid(outputs))
    predicted = outputs.data >= 0

    predicted = predicted.type_as(labels)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    
print('Accuracy of the model on the test images: %f %%' % (100 * (correct / total)))