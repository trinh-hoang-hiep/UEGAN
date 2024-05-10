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




dataset_list = ['CAMO']
root = "/home/hoang/Desktop/luanvan/data/CAMO-V.1.0_CVIU2019-20220314T013357Z-001/GTtest"
for dataset in dataset_list:    
    
    name_list = sorted(os.listdir(root))
    print('[INFO]: Process [{}]'.format(dataset))

    nlist= ["camourflage_00470.png","camourflage_01170.png","camourflage_00079.png","camourflage_01190.png","camourflage_01185.png","camourflage_01217.png","camourflage_00398.png"]

    for name in nlist:
        img_list = []
        for i in range(10):
            img_root = os.path.join('/home/hoang/Downloads/ecetest/79_epochsotajac/79_epoch_{}/CAMO'.format( i), name)
            img = cv2.imread(img_root).astype(np.float64) / 255.0
            img_list.append(img)
        predictive_norm_color = calculate(img_list)
        save_root = os.path.join('./uncertainty_CAMO_UEGAN')
        os.makedirs(save_root, exist_ok=True)
        cv2.imwrite(os.path.join(save_root, img_root.split('/')[-1]), predictive_norm_color)
    































































