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

from test_uncertainty import ece_binary,UncertaintyAndCorrectionEvalNumpy, UncertaintyErrorDiceNumpy
import math
import numpyfunctions as np_fn 

def add_background_probability(probability_np: np.ndarray):
    probability_with_background = np.stack([1 - probability_np, probability_np], axis=-1)
    return probability_with_background

def Uentropy(logits,c):
    predictive = np_fn.entropy(logits) / np.log(2)
    
    return predictive


def cal_ueo(to_evaluate,thresholds):
    UEO = []
    for threshold in thresholds:
        results = dict()
        
        metric = UncertaintyErrorDiceNumpy(threshold)
        metric(to_evaluate,results)
        ueo = results['corrected_add_dice']
        UEO.append(ueo)
    max_UEO = max(UEO)
    return max_UEO


trans = transforms.Compose([transforms.ToTensor()])
def get_img_pil(path):
    img = Image.open(path).convert('L')
    
    gt=Image.open(root+'/'+path.replace('jpg','png').split("/")[-1]).convert('L')
    img = img.resize(gt.size, Image.BILINEAR)
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

dataset_list = ['CAMO']
root = "/home/hoang/Desktop/luanvan/data/CAMO-V.1.0_CVIU2019-20220314T013357Z-001/GTtest"

thresholds=[0.4]
for threshold in thresholds:
    print("threshold is:",threshold)
    for dataset in dataset_list:    
        
        name_list = sorted(os.listdir(root))
        print('[INFO]: Process [{}]'.format(dataset))
        ueo_cacanh=[]
        tongpixel=[]
        for name in tqdm(name_list):
            img_list = []
            for i in range(1):
                img_root = os.path.join('/home/hoang/Downloads/ecetest/uefinetune9/06_epoch_{}/CAMO'.format( i), name)
                # img_root = os.path.join('/home/hoang/Downloads/ecetest/phuongphapkhac/evpv2/CAMO'.format( i), name.replace("png","jpg"))
                
                img = get_img_pil(img_root)
                img=trans(img)
                img=np.array(img)

                img_list.append(img)
            

            gt_root=os.path.join(root, name)
            gt = get_img_pil(gt_root)
            gt=trans(gt)
            gt=np.array(gt)

            
            logit=np.array(sum(img_list)/1.0)

            prob=add_background_probability(logit)
            uncertainty = Uentropy(prob, 2)
            uncertainty = (uncertainty-uncertainty.min()) / (uncertainty.max()-uncertainty.min())
            
            thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

            to_evaluate = dict()
            to_evaluate['target'] = gt 
            
            U=uncertainty
            to_evaluate['prediction'] = logit
            to_evaluate['uncertainty'] = U
            array_row=[]
            for thres in range(255,0,-1):
                thres=thres/255.0
                prediction=to_evaluate['prediction'] 
                prediction[to_evaluate['prediction']>=thres]=1 
                prediction[to_evaluate['prediction']<thres]=0

                to_evaluate['prediction']=prediction
                UEO = cal_ueo(to_evaluate, thresholds)
                array_row.append(UEO)

            ueo_cacanh.append(array_row)
        ueotong=np.array(ueo_cacanh)
        ueotong=np.mean(ueotong, axis=0)
        print(ueotong.shape)
        print(np.mean(ueotong))
