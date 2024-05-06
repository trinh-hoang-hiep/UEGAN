import torch.utils.data as data
from dataset.dataloader import SalObjDatasetRGBD, SalObjDatasetWeak, SalObjDatasetRGB


def get_loader(option, pin_memory=True):
    if option['task'] == 'RGBD-SOD': #7
        dataset = SalObjDatasetRGBD(option['paths']['image_root'], option['paths']['gt_root'], 
                                    option['paths']['depth_root'], trainsize=option['trainsize'])
    elif option['task'] == 'Weak-RGB-SOD': #7
        dataset = SalObjDatasetWeak(option['paths']['image_root'], option['paths']['gt_root'], 
                                    option['paths']['mask_root'], option['paths']['gray_root'], 
                                    trainsize=option['trainsize'])
    else:
        dataset = SalObjDatasetRGB(option['paths']['image_root'], option['paths']['gt_root'], trainsize=option['trainsize']) #7
    data_loader = data.DataLoader(dataset=dataset, #7
                                  batch_size=option['batch_size'],
                                  shuffle=True,
                                  num_workers=option['batch_size'],
                                  pin_memory=pin_memory) #7
    return data_loader, dataset.size #7
