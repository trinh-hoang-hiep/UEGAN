from tqdm import tqdm
import numpy as np
import cv2
import os


import cv2
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
import numpy as np
import pdb, os, argparse

import os
import torch
import torch.utils.data as data
import random
import numpy as np
from PIL import Image
from PIL import ImageEnhance
from numpyfunctions import *

def ece_score(img_list,confidence,y_test, n_bins=10):
        py=sum(img_list)/10.0
        acc, conf = np.zeros(n_bins), np.zeros(n_bins)
        Bm = np.zeros(n_bins)
        py = np.array(py)
        y_test = np.array(y_test)
        
        py,y_test=py.reshape(1,-1),y_test.reshape(1,-1)
        confidence=confidence.reshape(1,-1)
        confidence=np.copy(py)

        gtt= np.copy(py)
        gtt[gtt > 0.5] = 1 
        gtt[gtt < 0.5] = 0

        accgoc=(gtt+y_test > 1.5)

        for m in range(n_bins):
            py_value= np.zeros(py.shape)
            
            py_value2= np.copy(confidence)
            
            
            a, b = m / n_bins, (m + 1) / n_bins
            
            if(a==0):
                py_value[(confidence>= a)& (confidence <= b)]=1
            else:
                py_value[(confidence> a)& (confidence <= b)]=1
            Bm[m] += py_value.sum()
            
            if(a==0):
                acc[m]+=accgoc[(confidence>= a)& (confidence <= b)].sum()
            else:
                acc[m]+=accgoc[(confidence> a)& (confidence <= b)].sum()
            

            py_value2[confidence <= a ]=0
            py_value2[confidence > b]=0
            conf[m] += py_value2.sum()

            if Bm[m] != 0:
                acc[m] = acc[m] / Bm[m]
                conf[m] = conf[m] / Bm[m]
        ece = 0
        for m in range(n_bins):
            ece += Bm[m] * np.abs((acc[m] - conf[m]))
            
        ece= ece / sum(Bm)
        return ece



trans = transforms.Compose([transforms.ToTensor()])
def get_img_pil(path):
    img = Image.open(path).convert('L')
    return img

def calculate(img_list):
    mean = np.zeros_like(img_list[0])
    for img in img_list:
        mean += img
    mean = mean / len(img_list)
    predictive = -mean * np.log(np.minimum(mean+1e-8, 1))
    predictive_norm = (predictive-predictive.min()) / (predictive.max()-predictive.min())
    
    

    mean[mean<0.5]=1-mean[mean<0.5] 
    
    return mean


dataset_list = ['Kvasir','CVC-ClinicDB','CVC-ColonDB','ETIS-LaribPolypDB','CVC-300'] 

for dataset in dataset_list:    
    root = "/home/hoang/Downloads/ecetest/GTmedical/TestDataset/{}/masks".format(dataset) 
    
    name_list = sorted(os.listdir(root))
    print('[INFO]: Process [{}]'.format(dataset))
    ece_cacanh=[]
    tongpixel=[]
    for name in name_list: 
        img_list = []
        for i in range(1):
            img_root = os.path.join('/home/hoang/Downloads/ecetest/ieeemedical/38_epoch_0iGAN/{}'.format( dataset), name) 
            
            img = get_img_pil(img_root)
            img=trans(img)
            img=np.array(img)

            img_list.append(img)
        

        gt_root=os.path.join(root, name)
        gt = get_img_pil(gt_root)
        gt=trans(gt)
        gt=np.array(gt)
        
        
        ece= ece_binary(np.array(sum(img_list)/1.0),np.array(gt), n_bins=10)
    
    
        ece_cacanh.append(ece[0])
        tongpixel.append(ece[1])
    
    ecetong=sum(ece_cacanh)/sum(tongpixel)
    
    print(ecetong)

