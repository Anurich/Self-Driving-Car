import torch
import pandas as pd
import torchvision
import numpy as np
import cv2
import torch.nn as nn
import os
from PIL import Image
# read the data

class dataset(torch.utils.data.Dataset):
    def __init__(self,transform=None):
        super(dataset,self).__init__()
        self.all_data =None
        self.imageFolder = "data/"
        with open("data.txt","r") as fp:
            self.all_data =  fp.readlines()
        self.imageName = []
        self.steeringAngle = []
        for line in self.all_data:
            x,y = line.split(",")[0].split()
            self.imageName.append(x)
            self.steeringAngle.append(np.float32(y))

        self.min = np.min(self.steeringAngle)
        self.max = np.max(self.steeringAngle)
        self.newSteeringAngle = (self.steeringAngle - self.min)/(self.max - self.min)
        '''self.newSteeringAngle = []
        for angle in self.steeringAngle:
            if angle < -0.25:
                angle += 0.25
            elif angle > 0.25:
                angle -= 0.25
            self.newSteeringAngle.append(angle)
        '''
        self.transform = transform
    def __getitem__(self,idx):
        image = self.imageName[idx]
        steering = self.newSteeringAngle[idx]
        #read the image and prprocess it
        img = cv2.imread(os.path.join(self.imageFolder,image))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if self.transform != None:
            img  = self.transform(img)
        return img,steering
    def __len__(self):
        return len(self.all_data)

