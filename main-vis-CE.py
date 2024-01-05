'''
Created on Apr 24, 2022

@author: deckyal
'''
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from config import device
import os

from torch.utils.tensorboard import SummaryWriter ##

from dataset import *
from models import *
from utils import * 
from metrics import * 
from _operator import truediv

def train(model = None, train_loader = None, val_loader=None, optimizer = None, SavingName=None, writer = None, tLoss=None):
    
    total_step = len(train_loader)
    best_acc = -9999
    acc = 0.01
    
    bestModelName =  SavingName+'-best-'+str(0)+'.ckpt'
    
    for epoch in range(num_epochs):
        averageLoss = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            #loss = torch.sqrt(torch.mean((outputs - labels) ** 2))
            output = tLoss(outputs, labels.squeeze().long())

            optimizer.zero_grad()
            output.backward()
            optimizer.step()
            
            tl = output.detach().cpu().numpy()
            
            averageLoss =+ tl
                      
            if (i + 1) % 2 == 0:
                message = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, tl)
                print(message)
        
        averageLoss = truediv(averageLoss, total_step) 
        writer.add_scalar('Loss/train-CE', averageLoss,epoch)##
        writer.add_scalar('acc/val-CE', acc,epoch)##
        
        #do validations every 10 epoch 
        if epoch%10 == 0:
            with torch.no_grad():
                
                model.eval()        
                pred,gt = [],[]
                
                for imagesV, labelsV in val_loader:
                    
                    imagesV = imagesV.to(device)
                    labelsV = labelsV.to(device)
                    
                    # Forward pass
                    outputsV =  torch.argmax(model(imagesV), dim=1)
                    
                    gt.extend(labelsV.squeeze().cpu().numpy())
                    pred.extend(outputsV.squeeze().cpu().numpy())
                
                gt = np.asarray(gt,np.float32)
                pred = np.asarray(pred)
                acc = accuracy(pred,gt)
        
                print('Val Accuracy of the model on the {} epoch: {} %'.format(i,acc))
                
            model.train()
    
    # Save the model checkpoint
    checkDirMake(os.path.dirname(SavingName))
    torch.save(model.state_dict(), SavingName)
    # to load : model.load_state_dict(torch.load(save_name_ori))
    
def test(model = None,SavingName=None, test_loader=None, writer = None):
    # Test the model
    model.load_state_dict(torch.load(SavingName))

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
         
        pred,gt = [],[]
        
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = torch.argmax(model(images), dim=1)
            
            gt.extend(labels.squeeze().cpu().numpy())
            pred.extend(outputs.squeeze().cpu().numpy())
        
        gt = np.asarray(gt,np.float32)
        pred = np.asarray(pred)
        
        print('Test Accuracy of the model on test images: {} %'.format(accuracy(pred,gt)))
        
        if writer is not None: 
            writer.add_scalar('acc/test-CE', accuracy(pred,gt),0)##
        
        
if __name__ == '__main__':
        
    tr = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
        ])
    
    
    batch_size = 16 
    
    CDTrain = CatDog('../../../Data/IC-CatDog/train-small/', transform=tr,crossNum=5, crossIDs=[1,2,3,4])
    train_loader = torch.utils.data.DataLoader(dataset=CDTrain,
                                               batch_size=batch_size,
                                               shuffle=False)
    
    CDVal = CatDog('../../../Data/IC-CatDog/train-small/', transform=tr,crossNum=5, crossIDs=[5])
    val_loader = torch.utils.data.DataLoader(dataset=CDVal,
                                               batch_size=batch_size,
                                               shuffle=False)
    
    CDTest = CatDog('../../../Data/IC-CatDog/test-small/', tr)
    test_loader = torch.utils.data.DataLoader(dataset=CDTest,
                                              batch_size=16,
                                              shuffle=False)
    
    CNNI = CNN(num_classes = 2).to(device)
    loss = torch.nn.CrossEntropyLoss()
    
    if True: 
        from torchsummary import summary
        summary(CNNI,(3,224,224))
        print(CNNI)
    # input is the predictions in the shape of [b,c] b = batch c = class
    # input 2 is the label, can bein the shape of [b], or softmax of [b,c] 
    
    savingName = './checkpoints/CNN-CatDog-CE'
    
    learning_rate = .0001
    num_epochs =200
    optimizer = torch.optim.Adam(CNNI.parameters(), lr=learning_rate)



    writer = SummaryWriter()
    operation = 2
    
    
    if operation ==0 or operation==2: 
        train(model = CNNI, train_loader = train_loader, val_loader=val_loader, optimizer = optimizer, 
                          SavingName = savingName, writer = writer,tLoss = loss)
    if operation ==1 or operation==2: 
        test(model = CNNI,SavingName = savingName, test_loader=test_loader, writer = writer)
    
    
    writer.close()
        
        
        