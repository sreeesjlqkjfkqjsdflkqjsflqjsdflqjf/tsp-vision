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
    def __init__(self, dataDir='./leapGestRecog/', transform=None, crossNum=None, crossIDs=None):
        # Initialize the data and label list
        self.labels = []
        self.data = []
        temp = []
        tempL = []

        # First load all images data
        import os
        liste_sujet = os.listdir(dataDir)
        liste_geste = [[dataDir + i + '/' + j for j in os.listdir(dataDir + i)] for i in liste_sujet]
        liste_image = []
        liste_label = []
        for sujet in liste_geste:
            for geste in sujet:
                for image in os.listdir(geste):
                    liste_image.append(geste + '/' + image)
                    liste_label.append(geste)
        if transform is not None:
            self.transform = transform
        for x in tqdm(liste_image):
            # Read the data using PIL
            image = Image.open(x).convert('RGB')
            if self.transform is not None:
                image_tensor = self.transform(image)
            temp.append(image_tensor)

            # Second filter according name for labelling : 1 = palm 02=l 03=fist 04=fist_moved 05=thumb  06=index 07=ok
            # 08=palm_moved 09=c 10=down

            # Get label in the path and convert it
            label_path = liste_label[liste_image.index(x)]
            label_path_splitted = label_path.split("/")
            label = label_path_splitted[3][:2]
            label = int(label) - 1
            # Create a one hot tensor with 1 in the corresponding index
            label_tensor = torch.zeros(10)
            label_tensor[label] = 1
            # Append
            tempL.append(label_tensor)

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
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),  
        transforms.RandomRotation(15),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizing mean and std values
    ])
    cd = GestRecog(transform=tr, crossNum=None, crossIDs=None)
    print(len(cd))
    image, label = next(iter(cd))
    torch.save(cd.data, 'images.pth')
    torch.save(cd.labels, 'labels.pth')
    exit(0)
