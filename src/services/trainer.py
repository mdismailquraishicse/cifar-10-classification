import torch
import torch.nn as nn
from torch.optim import Adam
from db.data_loader import CIFAR_10_LOADER
from services.cifar_10_classification import CIFAR10Net


class CIFAR10Trainer:


    def __init__(self):
        
        self.data_loader = CIFAR_10_LOADER()
        self.model = CIFAR10Net()


    def train(self, epochs:int=10, lr:float=0.01):
        
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(params=self.model.parameters(), lr=lr)
        for epoch in range(1, epochs+1):
            for i, (images, labels) in enumerate(self.data_loader.train_loader):
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            if (epoch % 3 == 0) or (epoch == epochs):
                print(f"Epoch: {epoch} Loss: {loss.item()}")