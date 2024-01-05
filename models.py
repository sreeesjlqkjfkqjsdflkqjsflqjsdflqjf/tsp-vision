'''
Created on Apr 25, 2022

@author: deckyal
'''

import torch
import torch.nn as nn

class FC(nn.Module):
    def __init__(self,inputNode=561,hiddenNode = 256, outputNode=1):   
        super(FC, self).__init__()     
        #Define Hyperparameters
        self.inputLayerSize = inputNode
        self.outputLayerSize = outputNode
        self.hiddenLayerSize = hiddenNode
        
        # weights
        self.Linear1 = nn.Linear(self.inputLayerSize, self.hiddenLayerSize)
        self.Linear2 = nn.Linear(self.hiddenLayerSize, self.outputLayerSize)
        
    def forward(self, X):
        
        self.z2 = self.Linear1(X) # 3 X 3 ".dot" does not broadcast in PyTorch
        self.a2 = self.sigmoid(self.z2) # activation function
        self.z3 = self.Linear2(self.a2)
        return self.z3
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+torch.exp(-z))
    
    def loss(self, yHat, y):
        J = 0.5*sum((y-yHat)**2)
        

class CNN(nn.Module):
    def __init__(self, num_classes=1):
        super(CNN, self).__init__()

        self.conv11 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.conv12 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear(56 * 56 * 64, num_classes)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        out11 = self.maxpool(self.relu(self.conv11(x)))
        
        #print(out11.shape)
        out12 = self.maxpool(self.relu(self.conv12(out11)))
        
        #print(out12.shape)
        
        out = out12.reshape(out12.size(0), -1)
        out = self.fc(out)

        return out
    
    

class CNN1D(nn.Module):
    def __init__(self, num_classes=1):
        super(CNN1D, self).__init__()

        self.conv11 = nn.Conv1d(1, 16, kernel_size=256, stride=1, padding=2)
        self.conv12 = nn.Conv1d(16, 32, kernel_size=128, stride=1, padding=1)

        self.fc = nn.Linear(32 * 2936, num_classes)
        #self.fc = nn.Linear(32 * 4, num_classes)

        self.maxpool = nn.MaxPool1d(kernel_size=16, stride=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        #print(x.shape)
        out11 = self.maxpool(self.relu(self.conv11(x)))
        
        #print(out11.shape)
        out12 = self.maxpool(self.relu(self.conv12(out11)))
        
        #print(out12.shape)
        
        out = out12.reshape(out12.size(0), -1)
        out = self.fc(out)

        return out