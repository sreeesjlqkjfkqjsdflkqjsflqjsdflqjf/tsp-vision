'''
Created on Apr 24, 2022

@author: deckyal
'''
import torch
from torchvision import transforms
from PIL import Image
from _operator import truediv
from torchvision.transforms import ToTensor
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


class GestRecog(torch.utils.data.Dataset):
    def __init__(self, dataDir='./Dataset_rps/', transform=None, crossNum=None, crossIDs=None, Data_augment=None):
        # Initialize the data and label list
        self.data_perso=[]
        self.labels_perso=[]
        temp = []
        tempL = []
        if transform is not None:
            self.transform = transform

        # First load all images data
        import os
        liste_classe = ["Scissors", "Rock", "Paper"]
        for n, i in enumerate(tqdm(liste_classe)):
            liste_image = [dataDir+i+"/"+j  for j in os.listdir(dataDir+i)]
            for chem_im in tqdm(liste_image, leave = False):
                image = Image.open(chem_im).convert('RGB')
                if self.transform is not None:
                    image_tensor = self.transform(image)
                self.data_perso.append(image_tensor)
                label_tensor = torch.zeros(len(liste_classe))
                label_tensor[n] = 1
                self.labels_perso.append(label_tensor)


        if Data_augment is not None:
            for i in tqdm(range(len(self.data_perso))):
                transformed_tensor=Data_augment(self.data_perso[i])
                self.data_perso.append(transformed_tensor)
                self.labels_perso.append(self.labels_perso[i])



        if crossNum is not None:
            totalLength = len(temp)
            length = int(truediv(totalLength, crossNum))

            for crossID in crossIDs:
                lowR = crossID - 1
                if crossID == crossNum:
                    self.data.extend(temp[(lowR) * length:])
                    self.labels.extend(tempL[(lowR) * length:])
                else:
                    self.data.extend(temp[(lowR) * length:(crossID) * length])
                    self.labels.extend(tempL[(lowR) * length:(crossID) * length])
        else:
            self.data = temp
            self.labels = tempL

        self.transform = transform

    def __getitem__(self, index):

        data = self.data[index]
        lbl = self.labels[index]

        return data, lbl

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    tr = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        #transforms.RandomHorizontalFlip(),  
        #transforms.RandomRotation(90),  
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
        transforms.Normalize(mean=[0], std=[1])  # Normalizing mean and std values
    ])
    tr_data_augment=transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
    ])
    cd = GestRecog(transform=tr, crossNum=None, crossIDs=None,Data_augment=tr_data_augment)
    torch.save(cd.data_perso,"images_perso.pth")
    torch.save(cd.labels_perso,"labels_perso.pth")
    exit(0)
