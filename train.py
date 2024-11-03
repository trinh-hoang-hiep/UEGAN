import os
import torch
from glob import glob
from dataset.get_loader import get_loader
from config import param as option
from utils import set_seed, save_scripts
from model.get_model import get_model
from loss.get_loss import get_loss
from optim.get_optim import get_optim, get_optim_dis
from trainer.get_trainer import get_trainer
from torch.utils.tensorboard import SummaryWriter
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

import torch
import torch.nn as nn
import torch.nn.functional as F
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
SAMPLES    = 3

if __name__ == "__main__":

    print('[INFO] Experiments saved in: ', option['training_info']) 
    set_seed(option['seed']) 
    train_one_epoch = get_trainer(option) 
    loss_fun = get_loss(option) 
    model, dis_model = get_model(option) 

    num_parameters = sum(parameter.numel() for parameter in model.parameters())

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
    for parameter in model.parameters():
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
        model,
        
        
        
        
        Normal,
        {"loc": normal_posterior_loc, "scale": normal_posterior_softplus_inv_scale},
        num_particles,
    )
    model = ParticlePosteriorModel(
        model,
        lambda weight, loc, log_scale: MixtureSameFamily(Categorical(weight), Independent(LogScaleNormal(loc, log_scale), 1)),
        {"weight": normal_mixture_prior_weight, "loc": normal_mixture_prior_loc, "log_scale": normal_mixture_prior_log_scale},
        
        
        
        
        num_particles,

    )
    for parameter ,parameter2 in zip(model.models.parameters(),model2.models.parameters()):
        parameter.data.copy_(parameter2.data)
    
    
    if option['checkpoint'] is not None: 
        model.load_state_dict(torch.load(option['checkpoint']))
        print('Load checkpoint from {}'.format(option['checkpoint']))
    kernel = RBFSteinKernel()
    
    preconditioner = SVGD(model.parameters(), kernel, num_particles)
    criterion = NLUPLoss()
   

    
    #print(model)
    optimizer, scheduler = get_optim(option, model.parameters()) 
    schedule = StepLR(optimizer, step_size=1, gamma=gamma)
    if dis_model is not None: 
        optimizer_dis, scheduler_dis = get_optim_dis(option, dis_model.parameters()) 
    else:
        optimizer_dis, scheduler_dis = None, None
    train_loader, dataset_size = get_loader(option) 
    model_list, optimizer_list = [model, dis_model], [optimizer, optimizer_dis] 
    writer = SummaryWriter(option['log_path']) 
    
    save_scripts(option['log_path'], scripts_to_save=glob('*.*')) 
    save_scripts(option['log_path'], scripts_to_save=glob('dataset/*.py', recursive=True)) 
    save_scripts(option['log_path'], scripts_to_save=glob('model/*.py', recursive=True)) 
    save_scripts(option['log_path'], scripts_to_save=glob('optim/*.py', recursive=True)) 
    save_scripts(option['log_path'], scripts_to_save=glob('trainer/*.py', recursive=True)) 
    save_scripts(option['log_path'], scripts_to_save=glob('model/blocks/*.py', recursive=True)) 
    save_scripts(option['log_path'], scripts_to_save=glob('model/backbone/*.py', recursive=True)) 
    save_scripts(option['log_path'], scripts_to_save=glob('model/decoder/*.py', recursive=True)) 
    save_scripts(option['log_path'], scripts_to_save=glob('model/depth_module/*.py', recursive=True)) 
    save_scripts(option['log_path'], scripts_to_save=glob('model/neck/*.py', recursive=True)) 

    for epoch in range(1, (option['epoch']+1)): 
        model_dict, loss_record = train_one_epoch(epoch, model_list, optimizer_list, train_loader, dataset_size, loss_fun,criterion,log_prior_weight,preconditioner ) 

        
        writer.add_scalar('loss', loss_record.show(), epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        scheduler.step()
        if scheduler_dis is not None:
            scheduler_dis.step()

        save_path = option['ckpt_save_path']

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if epoch % 1==0:
            for model_name in model_dict.keys():
                save_name = os.path.join(save_path, '{:0>2d}_{:.3f}_{}.pth'.format(epoch, loss_record.show(), model_name))
                if model_dict[model_name] is not None:
                    torch.save(model_dict[model_name].state_dict(), save_name)

        os.system("CUDA_VISIBLE_DEVICES=2 python  test.py  --uncer_method ganabp --ckpt last --task COD")

        
        
