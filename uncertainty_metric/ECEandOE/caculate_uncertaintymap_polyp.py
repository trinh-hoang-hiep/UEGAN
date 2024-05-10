
from tqdm import tqdm
import numpy as np
import cv2
import os


def calculate(img_list):
    mean = np.zeros_like(img_list[0])
    for img in img_list:
        mean += img
    mean = mean / len(img_list)
    predictive = -mean * np.log(np.minimum(mean+1e-8, 1))
    predictive_norm = (predictive-predictive.min()) / (predictive.max()-predictive.min())*255
    predictive_norm_color = cv2.applyColorMap(np.array(predictive_norm, np.uint8), cv2.COLORMAP_JET)
    
    return predictive_norm_color


dataset_list = ['Polyp']
root = "/home/hoang/Downloads/ecetest/GTmedical/TestDataset/ETIS-LaribPolypDB/masks"
for dataset in dataset_list:    
    
    name_list = sorted(os.listdir(root))
    print('[INFO]: Process [{}]'.format(dataset))


    nlist= ["188.png"]
    for name in nlist:
        
        
        img_list = []
        for i in range(10):
            img_root = os.path.join('/home/hoang/Downloads/ecetest/medi28/28_epoch_0/ETIS-LaribPolypDB', name)
            img = cv2.imread(img_root).astype(np.float64) / 255.0
            img_list.append(img)
        predictive_norm_color = calculate(img_list)
        save_root = os.path.join('./uncertainty_polyp_UEGAN')
        os.makedirs(save_root, exist_ok=True)
        cv2.imwrite(os.path.join(save_root, img_root.split('/')[-1]), predictive_norm_color)   
