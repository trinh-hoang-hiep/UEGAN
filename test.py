import cv2
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
import numpy as np
import pdb, os, argparse
from dataset.dataloader import test_dataset, eval_Dataset, test_dataset_rgbd
from tqdm import tqdm

from config import param as option
from model.get_model import get_model
from utils import sample_p_0, DotDict
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from torch.distributions import Categorical, Independent, MixtureSameFamily
from torch.distributions import Normal
from bayestorch.distributions import LogScaleNormal, SoftplusInvScaleNormal
from bayestorch.distributions import LogScaleNormal
from bayestorch.kernels import RBFSteinKernel
from bayestorch.losses import NLUPLoss
from bayestorch.models import ParticlePosteriorModel
from bayestorch.preconditioners import SVGD
SAMPLES=3



import torch
import torch.nn as nn
import torch.nn.functional as F
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
from PIL import Image 

def eval_f1(loader, cuda=True):
    predictions_SVM, Test_Y= [],[]
    with torch.no_grad():
        
        for pred, gt in loader:
            predictions_SVM.append(pred.tolist())
            Test_Y.append(gt.tolist())
        
    print("SVM f1 Score -> ",f1_score(predictions_SVM, Test_Y, average='macro')*100)
    return 0
def eval_mae(loader, cuda=True):
    avg_mae, img_num, total = 0.0, 0.0, 0.0
    with torch.no_grad():
        for pred, gt in loader:
            if cuda:
                pred, gt = pred.to(device), gt.to(device)
            else:
                pred, gt = (pred), (gt)
            mae = torch.abs(pred - gt).mean()
            if mae == mae: 
                avg_mae += mae
                img_num += 1.0
        avg_mae /= img_num
    return avg_mae


class Tester():
    def __init__(self, option):
        self.option = option 
        self.test_epoch_num = option['checkpoint'].split('/')[-1].split('_')[0] 
        self.model, self.uncertainty_model = get_model(option) 
        
        
        num_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        
        normal_mixture_prior_weight=0.75
        normal_mixture_prior_log_scale1=-1.0
        normal_mixture_prior_log_scale2=-6.0
        num_particles=SAMPLES
        log_prior_weight=1e-9
        gamma=0.7
        normal_mixture_prior_weight = torch.tensor(
            [normal_mixture_prior_weight, 1 - normal_mixture_prior_weight]
        )
        
        
        mitb5=torch.zeros( num_parameters)
        start_idx = 0
        for parameter in self.model.parameters():
            num_elements = parameter.numel()
            mitb5[
                start_idx : start_idx + num_elements
            ].copy_(torch.flatten(parameter.data))
            
            
            
            
            start_idx += num_elements


        normal_mixture_prior_loc =torch.stack(
            [mitb5, mitb5]
        )
        normal_mixture_prior_log_scale1 = torch.full((num_parameters,), normal_mixture_prior_log_scale1)
        normal_mixture_prior_log_scale2 = torch.full((num_parameters,), normal_mixture_prior_log_scale2)
        normal_mixture_prior_log_scale = torch.stack(
            [normal_mixture_prior_log_scale1, normal_mixture_prior_log_scale2]
        )


        def get_rho(sigma, delta):
            """
            sigma is represented by softplus function  'sigma = log(1 + exp(rho))' to make sure it 
            remains always positive and non-transformed 'rho' gets updated during backprop.
            """
            rho = torch.log(torch.expm1(delta * torch.abs(sigma)) + 1e-20)
            return rho
        
        normal_posterior_loc = mitb5
        normal_posterior_softplus_inv_scale =torch.log1p(torch.exp(get_rho(mitb5, 0.05)))
        
        model2 = ParticlePosteriorModel(
            self.model,
           
            Normal,
            {"loc": normal_posterior_loc, "scale": normal_posterior_softplus_inv_scale},
            num_particles,
        )
        self.model = ParticlePosteriorModel(
            self.model,
            lambda weight, loc, log_scale: MixtureSameFamily(Categorical(weight), Independent(LogScaleNormal(loc, log_scale), 1)),
            {"weight": normal_mixture_prior_weight, "loc": normal_mixture_prior_loc, "log_scale": normal_mixture_prior_log_scale},

            num_particles,

        )

        self.model.load_state_dict(torch.load(option['checkpoint']))
        self.model.eval()

        param_count = sum(x.numel()/1e6 for x in self.model.parameters()) 
        print("[INFO]: GENModel based on [{}] have {:.4f}Mb paramerters in total".format(option['model_name'], param_count)) 
        
        param_count = sum(x.numel()/1e6 for x in self.uncertainty_model.parameters()) 
        print("[INFO]: DISModel based on [{}] have {:.4f}Mb paramerters in total".format(option['model_name'], param_count)) 
    def prepare_test_params(self, dataset, iter):
        save_path = os.path.join(option['eval_save_path'], self.test_epoch_num+'_epoch_{}'.format(iter), dataset) 
        print('[INFO]: Save_path is', save_path) 
        if not os.path.exists(save_path):  
            os.makedirs(save_path)
        if self.option['task'] == 'SOD' or self.option['task'] == 'Weak-RGB-SOD':
            image_root = os.path.join(self.option['paths']['test_dataset_root'], 'Imgs', dataset)
            test_loader = test_dataset(image_root, option['testsize'])
        elif self.option['task'] == 'RGBD-SOD':
            image_root = os.path.join(self.option['paths']['test_dataset_root'], dataset, 'RGB')
            test_loader = test_dataset_rgbd(image_root, option['testsize'])
        elif self.option['task'] == 'COD' :
            image_root = os.path.join(self.option['paths']['test_dataset_root']) 
            test_loader = test_dataset(image_root, option['testsize']) 
        
        return {'save_path': save_path, 'test_loader': test_loader} 

    def forward_a_sample(self, image, HH, WW, depth=None):
        res = self.model.forward(img=image, depth=depth)['sal_pre'][-1]
        
        res = F.upsample(res, size=[WW, HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
        
        return res

    def forward_a_sample_gan(self, image, HH, WW, depth=None,coord=None,cell=None):
        z_noise = torch.randn(image.shape[0], self.option['latent_dim']).to(device)

        res,_ = self.model.forward(img=image, z=z_noise, depth=depth,coord=coord,cell=cell) 

        res=res['sal_pre'][-1].mean(dim=0)
        
        res = F.upsample(res, size=[WW, HH], mode='bilinear', align_corners=False) 
        res = res.sigmoid().data.cpu().numpy().squeeze() 
        res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8) 
        
        return res 

    def forward_a_sample_ebm(self, image, HH, WW):
        
        opt = DotDict()
        opt.ebm_out_dim = 1
        opt.ebm_middle_dim = 100
        opt.latent_dim = 32
        opt.e_init_sig = 1.0
        opt.e_l_steps = 5
        opt.e_l_step_size = 0.4
        opt.e_prior_sig = 1.0
        opt.g_l_steps = 5
        opt.g_llhd_sigma = 0.3
        opt.g_l_step_size = 0.1
        opt.e_energy_form = 'identity'
        
        z_e_0 = sample_p_0(image, opt)
        
        z_e_0 = torch.autograd.Variable(z_e_0)
        z = z_e_0.clone().detach()
        z.requires_grad = True
        for kk in range(opt.e_l_steps):
            en = self.uncertainty_model(z)
            z_grad = torch.autograd.grad(en.sum(), z)[0]
            z.data = z.data - 0.5 * opt.e_l_step_size * opt.e_l_step_size * (
                    z_grad + 1.0 / (opt.e_prior_sig * opt.e_prior_sig) * z.data)
            z.data += opt.e_l_step_size * torch.randn_like(z).data

        z_e_noise = z.detach()  
        res = self.model.forward(img=image, z=z_e_noise)[-1]
        res = F.upsample(res, size=[WW, HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
        
        return res

    def test_one_detaset(self, dataset, iter):
        test_params = self.prepare_test_params(dataset, iter) 
        test_loader, save_path = test_params['test_loader'], test_params['save_path'] 

        time_list = [] 
        for i in tqdm(range(test_loader.size), desc=dataset): 
            image, depth, HH, WW, name,coord,cell, gtsize = test_loader.load_data()  
            image = image.to(device)
            
            if depth is not None: depth = depth.to(device)
            
            start = time.time() 
            if self.option['uncer_method'] == 'vae' or self.option['uncer_method'] == 'basic': 
                res = self.forward_a_sample(image, HH, WW, depth)
            elif self.option['uncer_method'] == 'ebm':
                import pdb; pdb.set_trace()
                res = self.forward_a_sample_ebm(image, HH, WW, depth)
            elif self.option['uncer_method'] == 'gan' or self.option['uncer_method'] == 'ganabp' or self.option['uncer_method'] == 'abp': 
                coord=torch.unsqueeze(coord, 0)
                cell=torch.unsqueeze(cell, 0)
                res = self.forward_a_sample_gan(image, HH, WW, depth,coord=coord[ :,0:147456, :].to(device),cell=cell[ :,0:147456, :] ) 
            
            end = time.time() 
            time_list.append(end-start) 
            
            def get_img_pil( path):
                img = Image.open(path).convert('L')
                return img

            cv2.imwrite(os.path.join(save_path, name), res) 
            ress=get_img_pil(os.path.join(save_path, name))
            ress = ress.resize(gtsize.size, Image.BILINEAR)
            ress.save(os.path.join(save_path, name)) 

        print('[INFO] Avg. Time used in this sequence: {:.4f}s'.format(np.mean(time_list)))

    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
            
    




iters = 10 
tester = Tester(option=option) 
for dataset in option['datasets']: 
    for i in range(iters):
        tester.test_one_detaset(dataset=dataset, iter=i)


mae_list = []
test_epoch_num = option['checkpoint'].split('/')[-1].split('_')[0]
print('========== Begin to evaluate the saved masks ==========')
for dataset in tqdm(option['datasets']):
    if option['task'] == 'RGBD-SOD' or option['task'] == 'COD':
        gt_root = option['paths']['test_dataset_root'] + dataset + '/GT'
    else:
        gt_root = option['paths']['test_dataset_root'] + '/GT/' + dataset + '/'
    mae_single_dataset = []
    for i in range(iters):
        
        loader = eval_Dataset(os.path.join(option['eval_save_path'], '{}_epoch_{}'.format(test_epoch_num, i), dataset), gt_root)
        mae = eval_mae(loader=loader, cuda=True)
        
        mae_single_dataset.append(mae.item())
        
        
        
    
    mae_list.append(np.mean(mae_single_dataset))
    print("list mae la", mae_single_dataset)
    print(np.mean(mae_single_dataset))
    print(np.std(mae_single_dataset))

print('--------------- Results ---------------')
results = np.array(mae_list)
results = np.reshape(results, [1, len(results)])
mae_table = pd.DataFrame(data=results, columns=option['datasets'])

with open(os.path.join(option['eval_save_path'], 'results_{}_epoch.csv'.format(test_epoch_num)), 'w') as f:
    mae_table.to_csv(f, index=False, float_format="%.4f")
print(mae_table.to_string(index=False))
print('--------------- Results ---------------')