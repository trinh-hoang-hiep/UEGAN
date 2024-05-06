import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
from PIL import ImageEnhance
from dataset.augment import cv_random_flip_rgb, randomCrop_rgb, randomRotation_rgb
from dataset.augment import cv_random_flip_rgbd, randomCrop_rgbd, randomRotation_rgbd
from dataset.augment import cv_random_flip_weak, randomCrop_weak, randomRotation_weak, colorEnhance, randomGaussian, randomPeper
import util.boundary_modification as boundary_modification 
def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret
def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.view(1, -1).permute(1, 0) 
    return coord, rgb
def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))
import torch
import random

def reseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

class SalObjDatasetRGB(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize):
        self.trainsize = trainsize 
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')] 
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')] 
        self.images = sorted(self.images) 
        self.gts = sorted(self.gts) 
        self.filter_files() 
        self.size = len(self.images) 
        self.img_transform = transforms.Compose([ 
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()]) 
        
        seg_normalization = transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            )
        self.seg_transform = transforms.Compose([
            transforms.ToTensor(),
            seg_normalization,
        ])
        
        self.bilinear_dual_transform = transforms.Compose([
            transforms.RandomCrop((384, 384), pad_if_needed=True),
            transforms.RandomHorizontalFlip(),
        ])

        self.bilinear_dual_transform_im = transforms.Compose([
            transforms.RandomCrop((384, 384), pad_if_needed=True),
            transforms.RandomHorizontalFlip(),
        ])


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index]) 
        gt = self.binary_loader(self.gts[index]) 
        image, gt = cv_random_flip_rgb(image, gt) 
        
        image, gt = randomRotation_rgb(image, gt) 
        image = colorEnhance(image) 
        gt = randomPeper(gt) 
        iou_max = 1.0
        iou_min = 0.8
        seed = np.random.randint(2147483647)
        
        reseed(seed)
        image = self.bilinear_dual_transform_im(image)

        reseed(seed)
        gt = self.bilinear_dual_transform(gt)
        iou_target = np.random.rand()*(iou_max-iou_min) + iou_min
        seg = boundary_modification.modify_boundary((np.array(gt)>0.5).astype('uint8')*255, iou_target=iou_target)
        seg = self.seg_transform(seg)
        
        def detach_to_cpu(x):
            return x.detach().cpu()
        inv_seg_trans = transforms.Normalize(
            mean=[-0.5/0.5],
            std=[1/0.5])
        def tensor_to_numpy(image):
            image_np = (image.numpy() * 255).astype('uint8')
            return image_np
        def transpose_np(x):
            return np.transpose(x, [1,2,0])
        def tensor_to_seg(x):
            x = detach_to_cpu(x)
            x = inv_seg_trans(x)
            x = tensor_to_numpy(x)
            x = transpose_np(x)
            return x
        def tensor_to_gray_im(x):
            x = detach_to_cpu(x)
            x = tensor_to_numpy(x)
            x = transpose_np(x)
            return x
        inv_im_trans = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225])
        def tensor_to_im(x):
            x = detach_to_cpu(x)
            x = inv_im_trans(x)
            x = tensor_to_numpy(x)
            x = transpose_np(x)
            return x


        
        hr_coord, hr_rgb = to_pixel_samples(seg.contiguous())

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / seg.shape[-2] 
        cell[:, 1] *= 2 / seg.shape[-1]

        crop_lr = resize_fn(seg, seg.shape[-2]) 
        return  {'image': image, 'gt': gt, 'index': index, 'seg': seg ,'inp': crop_lr, 'coord': hr_coord, 'cell': cell, 'gtt': hr_rgb}

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f: 
            img = Image.open(f) 
            
            return img.convert('RGB').resize((self.trainsize, self.trainsize)) 

    def binary_loader(self, path):
        with open(path, 'rb') as f: 
            img = Image.open(f) 
            
            
            return img.convert('L').resize((self.trainsize, self.trainsize)) 


    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size 


class SalObjDatasetRGBD(data.Dataset):
    def __init__(self, image_root, gt_root, depth_root=None, trainsize=352):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                       or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth = self.binary_loader(self.depths[index])
        image, gt, depth = cv_random_flip_rgbd(image, gt, depth)
        image, gt, depth = randomCrop_rgbd(image, gt, depth)
        image, gt, depth = randomRotation_rgbd(image, gt, depth)
        image = colorEnhance(image)
        gt = randomPeper(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        depth = self.depths_transform(depth)

        return {'image': image, 'gt': gt, 'depth': depth, 'index': index}

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        gts = []
        depths = []
        for img_path, gt_path, depth_path in zip(self.images, self.gts, self.depths):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth = Image.open(depth_path)
            if img.size == gt.size and gt.size == depth.size:
                images.append(img_path)
                gts.append(gt_path)
                depths.append(depth_path)
        self.images = images
        self.gts = gts
        self.depths = depths

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB').resize((self.trainsize, self.trainsize))     

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L').resize((self.trainsize, self.trainsize))     

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size == depth.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h), Image.NEAREST)
        else:
            return img, gt, depth

    def __len__(self):
        return self.size


class SalObjDatasetWeak(data.Dataset):
    def __init__(self, image_root, gt_root, mask_root, gray_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.masks = [mask_root + f for f in os.listdir(mask_root) if f.endswith('.png')]
        self.grays = [gray_root + f for f in os.listdir(gray_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.masks = sorted(self.masks)
        self.grays = sorted(self.grays)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.gray_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        mask = self.binary_loader(self.masks[index])
        gray = self.binary_loader(self.grays[index])
        image, gt, mask, gray = cv_random_flip_weak(image, gt, mask, gray)
        image, gt, mask, gray = randomCrop_weak(image, gt, mask, gray)
        image, gt, mask, gray = randomRotation_weak(image, gt, mask, gray)
        image = colorEnhance(image)
        
        gt = randomPeper(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        mask = self.mask_transform(mask)
        gray = self.gray_transform(gray)

        return {'image': image, 'gt': gt, 'mask': mask, 'gray': gray, 'index': index}

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images, gts, masks, grays = [], [], [], []
        for img_path, gt_path, mask_path, gray_path in zip(self.images, self.gts, self.masks, self.grays):
            img, gt, mask, gray = Image.open(img_path), Image.open(gt_path), Image.open(mask_path), Image.open(gray_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
                masks.append(mask_path)
                grays.append(gray_path)
        self.images = images
        self.gts = gts
        self.masks = masks
        self.grays = grays

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB').resize((self.trainsize, self.trainsize))     

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L').resize((self.trainsize, self.trainsize))    

    def depth_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('I')

    def resize(self, img, gt, mask, gray):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), mask.resize((w, h), Image.NEAREST), gray.resize((w, h), Image.NEAREST)
        else:
            return img, gt, mask, gray


    def __len__(self):
        return self.size


class test_dataset:
    def __init__(self, image_root, testsize):
        self.testsize = testsize 
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')] 


        self.images = sorted(self.images) 
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
        self.size = len(self.images) 
        self.index = 0 
        self.trainsize=testsize
        self.seg_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            ),
        ])
    
    def get_img_pil(self, path):
        img = Image.open(path).convert('L')
        return img   
    def load_data(self):
        image = self.rgb_loader(self.images[self.index]) 
        HH = image.size[0] 
        WW = image.size[1] 
        image = self.transform(image).unsqueeze(0) 
        name = self.images[self.index].split('/')[-1] 
        if name.endswith('.jpg'): 
            name = name.split('.jpg')[0] + '.png' 
        
        
        
        
        seg = self.binary_loader(self.images[self.index])

        seg = self.seg_transform(seg)
        
        hr_coord, hr_rgb = to_pixel_samples(seg.contiguous())

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / seg.shape[-2] 
        cell[:, 1] *= 2 / seg.shape[-1]
        crop_lr = resize_fn(seg, seg.shape[-2]) 
        gtsize=self.get_img_pil(self.images[self.index])
        self.index += 1 
        
        return image, None, HH, WW, name , hr_coord,cell, gtsize 


    def rgb_loader(self, path):
        with open(path, 'rb') as f: 
            img = Image.open(f) 
            return img.convert('RGB').resize((self.trainsize, self.trainsize)) 

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L').resize((self.trainsize, self.trainsize)) 


class test_dataset_rgbd:
    def __init__(self, image_root, testsize):
        depth_root = image_root[:-3] + 'depth'
        self.testsize = testsize
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.depths = [os.path.join(depth_root, f) for f in os.listdir(depth_root) if f.endswith('.bmp')
                       or f.endswith('.png')]
        self.images = sorted(self.images)
        self.depths = sorted(self.depths)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        
        
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.testsize, self.testsize)), transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0
        
        self.trainsize=testsize
        seg_normalization = transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            )
        self.seg_transform = transforms.Compose([
            transforms.ToTensor(),
            seg_normalization,
        ])
        
        self.bilinear_dual_transform = transforms.Compose([
            transforms.RandomCrop((384, 384), pad_if_needed=True),
            transforms.RandomHorizontalFlip(),
        ])

        self.bilinear_dual_transform_im = transforms.Compose([
            transforms.RandomCrop((384, 384), pad_if_needed=True),
            transforms.RandomHorizontalFlip(),
        ])

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        HH = image.size[0]
        WW = image.size[1]
        image = self.transform(image).unsqueeze(0)
        depth = self.binary_loader(self.depths[self.index])
        depth = self.depths_transform(depth).unsqueeze(0)

        name = self.images[self.index].split('/')[-1]
        
        
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size

        seg = self.seg_transform(seg)
        seg = self.binary_loader(self.images[self.index])

        
        hr_coord, hr_rgb = to_pixel_samples(seg.contiguous())

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / seg.shape[-2] 
        cell[:, 1] *= 2 / seg.shape[-1]
        return image, depth, HH, WW, name, hr_coord,cell

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB').resize((self.trainsize, self.trainsize))     

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L').resize((self.trainsize, self.trainsize))     

    def __len__(self):
        return self.size
       

class eval_Dataset(data.Dataset):
    def __init__(self, img_root, label_root):
        lst_label = sorted(os.listdir(label_root))
        
        lst_pred = sorted(os.listdir(img_root))
        
        self.label_abbr, self.pred_abbr = lst_label[0].split('.')[-1], lst_pred[0].split('.')[-1]
        label_list, pred_list = [], []
        for name in lst_label:
            label_name = name.split('.')[0]
            if label_name+'.'+self.label_abbr in lst_label:
                label_list.append(name)
    
        for name in lst_pred:
            label_name = name.split('.')[0]
            if label_name+'.'+self.pred_abbr in lst_pred:
                pred_list.append(name)

        self.image_path = list(map(lambda x: os.path.join(img_root, x), pred_list))
        self.label_path = list(map(lambda x: os.path.join(label_root, x), label_list))
        self.trans = transforms.Compose([transforms.ToTensor()])

    def get_img_pil(self, path):
        img = Image.open(path).convert('L')
        return img

    def __getitem__(self, item):
        img_path = self.image_path[item]
        label_path = self.label_path[item]
        pred = self.get_img_pil(img_path)  
        gt = self.get_img_pil(label_path)
        if pred.size != gt.size:
            
            
            pred = pred.resize(gt.size, Image.BILINEAR)
            

        return self.trans(pred), self.trans(gt)

    def __len__(self):
        return len(self.image_path)
    
    
class eval_Dataset_with_name(data.Dataset):
    def __init__(self, img_root_t, img_root_c, label_root):
        lst_label, lst_pred_t, lst_pred_c = sorted(os.listdir(label_root)), sorted(os.listdir(img_root_t)), sorted(os.listdir(img_root_c))
        self.label_abbr, self.pred_abbr_t, self.pred_abbr_c = lst_label[0].split('.')[-1], lst_pred_t[0].split('.')[-1], lst_pred_c[0].split('.')[-1]
        label_list, pred_t_list, pred_c_list = [], [], []
        for name in lst_label:
            label_name = name.split('.')[0]
            if label_name+'.'+self.label_abbr in lst_label:
                label_list.append(name)
    
        for name in lst_pred_t:
            label_name = name.split('.')[0]
            if label_name+'.'+self.pred_abbr_t in lst_pred_t:
                pred_t_list.append(name)
                
        for name in lst_pred_t:
            label_name = name.split('.')[0]
            if label_name+'.'+self.pred_abbr_c in lst_pred_t:
                pred_c_list.append(name)

        self.image_t_path = list(map(lambda x: os.path.join(img_root_t, x), pred_t_list))
        self.image_c_path = list(map(lambda x: os.path.join(img_root_c, x), pred_c_list))
        self.label_path = list(map(lambda x: os.path.join(label_root, x), label_list))
        self.trans = transforms.Compose([transforms.ToTensor()])

    def get_img_pil(self, path):
        img = Image.open(path).convert('L')
        return img

    def __getitem__(self, item):
        img_t_path = self.image_t_path[item]
        img_c_path = self.image_c_path[item]
        label_path = self.label_path[item]
        pred_t = self.get_img_pil(img_t_path)  
        pred_c = self.get_img_pil(img_c_path)  
        gt = self.get_img_pil(label_path)
        if pred_t.size != gt.size:
            pred_t = pred_t.resize(gt.size, Image.BILINEAR)
            pred_c = pred_c.resize(gt.size, Image.BILINEAR)
        name = '{}/{}'.format(img_t_path.split('/')[-2], img_t_path.split('/')[-1])

        return self.trans(pred_t), self.trans(pred_c), self.trans(gt), name

    def __len__(self):
        return len(self.image_t_path)