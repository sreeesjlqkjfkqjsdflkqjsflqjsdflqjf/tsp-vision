'''
Created on Apr 24, 2022

@author: deckyal
'''
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

from dataset import *
from models import *
from utils import * 
from metrics import * 
from config import device
import os

def train(model = None,SavingName=None, train_loader = None, val_loader=None, optimizer = None):
    # training
    print('training')
    
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            loss = torch.sqrt(torch.mean((outputs - labels) ** 2))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 2 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                
            
            #do validations every 10 epoch 
            if i%10 == 0:
                with torch.no_grad():
                    
                    model.eval()        
                    pred,gt = [],[]
                    
                    for imagesV, labelsV in val_loader:
                        
                        imagesV = imagesV.to(device)
                        labelsV = labelsV.to(device)
                        
                        # Forward pass
                        outputsV = model(imagesV).round()
                        
                        gt.extend(labelsV.squeeze().cpu().numpy())
                        pred.extend(outputsV.squeeze().cpu().numpy())
                    
                    gt = np.asarray(gt,np.float32)
                    pred = np.asarray(pred)
            
                    print('Val Accuracy of the model on the {} epoch: {} %'.format(i,accuracy(pred,gt)))
                    
                model.train()

    # Save the model checkpoint
    checkDirMake(os.path.dirname(SavingName))
    torch.save(model.state_dict(), SavingName)
    # to load : model.load_state_dict(torch.load(save_name_ori))
    
def test(model = None,SavingName=None, test_loader=None):
    # Test the model
    model.load_state_dict(torch.load(SavingName))

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
         
        pred,gt = [],[]
        
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).round()
            
            gt.extend(labels.squeeze().cpu().numpy())
            pred.extend(outputs.squeeze().cpu().numpy())
        
        gt = np.asarray(gt,np.float32)
        pred = np.asarray(pred)

        print('Test Accuracy of the model on test images: {} %'.format(accuracy(pred,gt)))
        
        
if __name__ == '__main__':
    
    
    tr = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
        ])
    
    
    batch_size = 16
    
    CDTrain = CatDog('../../../Data/IC-CatDog/train-small/', transform=tr,crossNum=5, crossIDs=[1,2,3,4])
    train_loader = torch.utils.data.DataLoader(dataset=CDTrain,
                                               batch_size=batch_size,
                                               shuffle=True)
    
    CDVal = CatDog('../../../Data/IC-CatDog/train-small/', transform=tr,crossNum=5, crossIDs=[5])
    val_loader = torch.utils.data.DataLoader(dataset=CDVal,
                                               batch_size=batch_size,
                                               shuffle=True)
    
    CDTest = CatDog('../../../Data/IC-CatDog/test-small/', transform=tr)
    test_loader = torch.utils.data.DataLoader(dataset=CDTest,
                                              batch_size=batch_size,
                                              shuffle=False)
    
    
    #FCI = FC(inputNode=3*64*64).to(device)
    CNNI = CNN().to(device)
    
    learning_rate = .0001
    num_epochs = 200
    optimizer = torch.optim.Adam(CNNI.parameters(), lr=learning_rate)



    operation = 2
    
    
    if operation ==0 or operation==2: 
        train(model = CNNI,SavingName='./checkpoints/CNN-CatDog-Baseline.ckpt', train_loader = train_loader, val_loader=val_loader, optimizer = optimizer)
    if operation ==1 or operation==2: 
        test(model = CNNI,SavingName='./checkpoints/CNN-CatDog-Baseline.ckpt', test_loader=test_loader)
        
        
        