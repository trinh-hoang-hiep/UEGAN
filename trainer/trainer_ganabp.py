import gc
import os
import pdb
import cv2
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from config import param as option
from utils import AvgMeter, label_edge_prediction, visualize_list, make_dis_label
from loss.get_loss import cal_loss
from utils import DotDict
if torch.cuda.is_available():########cal_loss#####################3
  device = torch.device("cuda")
else:  
  device = torch.device("cpu")

CE = torch.nn.BCELoss()
import numpy as np
import math
from copy import deepcopy
from scipy.spatial.distance import pdist, squareform



class JacobianReg(torch.nn.Module):

    def __init__(self, n=-1):
        #self.grad_v = grad_v
        assert n == -1 or n > 0
        self.n = n

        super(JacobianReg, self).__init__()

    def forward(self, x, y):
        '''
        computes (1/2) tr |dy/dx|^2
        '''
        y=y.view(y.shape[0],-1)
        B,C = y.shape
        if self.n == -1:
            num_proj = C
        else:
            num_proj = self.n
        J2 = 0
        J2=[]
        for ii in range(1):
            if self.n == -1:
                # orthonormal vector, sequentially spanned
                v=torch.zeros_like(y)
                v[:,:]=1
            else:
                v = self._random_vector(C=C,B=B)
            if x.is_cuda:
                v = v.cuda()
            Jv = self._jacobian_vector_product(y, x, v, create_graph=True)


            J2.append(Jv.flatten(start_dim=1))########################################
        J2=torch.stack(J2, dim=2)####################################J2=torch.cat(J2, axis=1)######
        J2=J2.view(J2.shape[0],J2.shape[1],-1)#######################
        J3=torch.transpose(J2, 1, 2)#########################
        jac=torch.bmm(J3,J2) 
        try:
            cholesky_factor = torch.linalg.cholesky(jac)
        except:
            
            jac += torch.eye(J2.shape[-1]).to(device) * 1e-9
            cholesky_factor = torch.linalg.cholesky(jac)
        cholesky_diagonal = torch.diagonal(cholesky_factor, dim1=1, dim2=2)
        log_det_jtj = 2 * torch.sum(torch.log(cholesky_diagonal), dim=1, keepdim=True)
        log_det_jtj=torch.absolute(log_det_jtj/2.0)
        # print(log_det_jtj.mean())
        return log_det_jtj.mean()###################################R 

    def _random_vector(self, C, B):
        '''
        creates a random vector of dimension C with a norm of C^(1/2)
        (as needed for the projection formula to work)
        '''
        if C == 1: 
            return torch.ones(B)
        v=torch.randn(B,C)
        arxilirary_zero=torch.zeros(B,C)
        vnorm=torch.norm(v, 2, 1,True)
        v=torch.addcdiv(arxilirary_zero, 1.0, v, vnorm)
        return v
                                                                            
    def _jacobian_vector_product(self, y, x, v, create_graph=False): 
        '''
        Produce jacobian-vector product dy/dx dot v.
        Note that if you want to differentiate it,
        you need to make create_graph=True
        '''                                                            
        flat_y = y.reshape(-1)
        flat_v = v.reshape(-1)
        grad_x, = torch.autograd.grad(flat_y, x, flat_v, 
                                        retain_graph=True, 
                                        create_graph=create_graph)
        return grad_x

def train_one_epoch(epoch, model_list, optimizer_list, train_loader, dataset_size, loss_fun,criterion,log_prior_weight,preconditioner ):##################################svgd
    ## Setup abp params
    opt = DotDict() #8
    opt.latent_dim = option['ganabp_config']['latent_dim'] #8
    opt.langevin_step_num_gen = option['ganabp_config']['step_num'] #8
    opt.sigma_gen = option['ganabp_config']['sigma_gen'] #8
    opt.langevin_s = option['ganabp_config']['langevin_s'] #8
    opt.pred_label = option['ganabp_config']['pred_label'] #8
    opt.gt_label = option['ganabp_config']['gt_label'] #8
    opt.lamda_dis = option['ganabp_config']['lamda_dis'] #8
    train_z = torch.FloatTensor(dataset_size, opt.latent_dim).normal_(0, 1).to(device) #8
    ## Setup abp params

    generator, discriminator = model_list #8
    generator_optimizer, discriminator_optimizer = optimizer_list #8
    generator.train() #8
    if discriminator is not None: #8
        discriminator.train() #8
    loss_record, supervised_loss_record, dis_loss_record = AvgMeter(), AvgMeter(), AvgMeter() #8
    print('Learning Rate: {:.2e}'.format(generator_optimizer.param_groups[0]['lr'])) #8
    progress_bar = tqdm(train_loader, desc='Epoch[{:03d}/{:03d}]'.format(epoch, option['epoch'])) #8
    for i, pack in enumerate(progress_bar): #8
        for rate in option['size_rates']: #10
            generator_optimizer.zero_grad() #10
            if discriminator is not None: #10
                discriminator_optimizer.zero_grad() #10
            if len(pack) == 3: ################################3 #10
                images, gts, depth, index = pack['image'].to(device), pack['gt'].to(device), None, pack['index'] #10
            elif len(pack) == 4:
                images, gts, depth, index = pack['image'].to(device), pack['gt'].to(device), pack['depth'].to(device), pack['index']
            elif len(pack) == 5:
                images, gts, mask, gray, depth = pack['image'].to(device), pack['gt'].to(device), pack['mask'].to(device), pack['gray'].to(device), None
            #############################  
            if len(pack) == 8: #####
                images, gts, depth, index = pack['image'].to(device), pack['gt'].to(device), None, pack['index'] #10
            # multi-scale training samples
            trainsize = (int(round(option['trainsize'] * rate / 32) * 32), int(round(option['trainsize'] * rate / 32) * 32)) #10
            if rate != 1: #10
                images = F.upsample(images, size=trainsize, mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=trainsize, mode='bilinear', align_corners=True)

            images.requires_grad_()###########################jacobian
            z_noise = torch.randn(images.shape[0], opt.latent_dim).to(images.device) #10
            z_noise = Variable(z_noise, requires_grad=True) #10
            z_noise_preds = [z_noise.clone() for _ in range(opt.langevin_step_num_gen + 1)] #10
            for kk in range(opt.langevin_step_num_gen): #10 #14
                z_noise = Variable(z_noise_preds[kk], requires_grad=True).to(device) #10 #14
                noise = torch.randn(z_noise.size()).to(device) #10 #14

                memory_chunk=images.shape[-2]*images.shape[-1]
                for i in range(0, images.shape[-2]*images.shape[-1], memory_chunk):###########################  32*32, 50176 = 224*224 , 384*384=147456
                    gen_res, log_priors = generator(img=images, z=z_noise, depth=depth,coord=pack['coord'][:, i:i+memory_chunk, :].to(device), cell=pack['cell'][:, i:i+memory_chunk, :]) #10 #14 #14... ###########################################svgd
                    gen_res=gen_res['sal_pre'][0].mean(dim=0)###############################svgd
                    gen_res=[gen_res]
                gen_loss = 0 #10 #14
                for i in gen_res: #10 #14 #14
                    if option['task'].lower() == 'weak-rgb-sod': #10 #14
                        gen_loss += 1 / (2.0 * opt.sigma_gen * opt.sigma_gen) * F.mse_loss(torch.sigmoid(i)*mask, gts*mask, size_average=True, reduction='sum')
                    else:
                        gen_loss += 1 / (2.0 * opt.sigma_gen * opt.sigma_gen) * F.mse_loss(torch.sigmoid(i), gts, size_average=True, reduction='sum') #10 #14
                if(epoch<=9):############################svgd
                    log_priors=1e-9*log_priors############################svgd
                    gen_loss-=log_priors.mean(0)##################################################################bayes  svgd cmt 
                gen_loss.backward(torch.ones(gen_loss.size()).to(device)) #10 #14

                grad = z_noise.grad #10 #14
                z_noise = z_noise + 0.5 * opt.langevin_s * opt.langevin_s * grad #10 #14
                z_noise += opt.langevin_s * noise #10 #14
                z_noise_preds[kk + 1] = z_noise #10 #14

            z_noise_post = z_noise_preds[-1]
            # pred_post = generator(img=images, z=z_noise_post, depth=depth)['sal_pre']
            for i in range(0, images.shape[-2]*images.shape[-1], memory_chunk):###########################   32*32, 50176 = 224*224
                # pred_post = generator(img=images, z=z_noise, depth=depth,coord=pack['coord'][:, i:i+memory_chunk, :].to(device), cell=pack['cell'][:, i:i+memory_chunk, :])['sal_pre']######################### 
                pred_post, log_priors = generator(img=images, z=z_noise_post, depth=depth,coord=pack['coord'][:, i:i+memory_chunk, :].to(device), cell=pack['cell'][:, i:i+memory_chunk, :])######################### svgd
                def clamp_probs(probs):
                    eps = torch.finfo(probs.dtype).eps
                    return probs.clamp(min=eps, max=1 - eps)
                ket_qua=torch.sigmoid(pred_post['sal_pre'][0])
                # print(ket_qua.shape) (torch.Size([3, 2, 1, 384]))
                ps_clamped = clamp_probs(ket_qua.mean(0))
                logp=torch.log(ps_clamped)#######################64,10
                min_real = torch.finfo(logp.dtype).min
                logp = torch.clamp(logp, min=min_real)
                p_log_p = ket_qua * logp#############################4,64,10
                entropy=-p_log_p.sum(0)
            pred_post=pred_post['sal_pre'][0].mean(dim=0)
            pred_post=[pred_post]

                

            if option['task'].lower() == 'sod':
                Dis_output = discriminator(torch.cat((images, torch.sigmoid(pred_post[0]).detach()), 1))
            elif option['task'].lower() == 'weak-rgb-sod':
                Dis_output = discriminator(torch.cat((images, mask*torch.sigmoid(pred_post[0]).detach()), 1))
            elif option['task'].lower() == 'cod':################################################
                Dis_output = discriminator(torch.cat((images, torch.sigmoid(pred_post[0]).detach()), 1))

            up_size = (images.shape[2], images.shape[3])
            Dis_output = F.upsample(Dis_output, size=up_size, mode='bilinear', align_corners=True)
            
            loss_dis_output = CE(torch.sigmoid(Dis_output), make_dis_label(opt.gt_label, gts))
            if (option['task'].lower() == 'sod') or (option['task'].lower() == 'cod'):##################################
                supervised_loss = cal_loss(pred_post, gts, loss_fun)
            elif option['task'].lower() == 'weak-rgb-sod':
                supervised_loss = loss_fun(images=images, outputs=pred_post, gt=gts, masks=mask, grays=gray, model=generator)
            # ############################
            loss_all = supervised_loss + opt.lamda_dis * loss_dis_output#####+uncertainty_ensemble_align*0.2#+0.01*loss_jr####*0.5###+0.1 * gal_loss##0.25##################################jacobian


            loss_all.backward()
            if(epoch<=40):
                preconditioner.step()############################# svgd 
            generator_optimizer.step()

            # train discriminator
            dis_pred = torch.sigmoid(pred_post[0]).detach()
            if  (option['task'].lower() == 'sod') or (option['task'].lower() == 'cod'):#########################################3
                Dis_output = discriminator(torch.cat((images, dis_pred), 1))
            elif option['task'].lower() == 'weak-rgb-sod':
                Dis_output = discriminator(torch.cat((images, mask*dis_pred), 1))

            Dis_target = discriminator(torch.cat((images, gts), 1))
            Dis_output = F.upsample(torch.sigmoid(Dis_output), size=up_size, mode='bilinear', align_corners=True)
            Dis_target = F.upsample(torch.sigmoid(Dis_target), size=up_size, mode='bilinear', align_corners=True)

            loss_dis_output = CE(torch.sigmoid(Dis_output), make_dis_label(opt.pred_label, gts))
            loss_dis_target = CE(torch.sigmoid(Dis_target), make_dis_label(opt.gt_label, gts))
            dis_loss = 0.5 * (loss_dis_output + loss_dis_target)
            dis_loss.backward()
            discriminator_optimizer.step()


            result_list = [torch.sigmoid(x) for x in pred_post]
            result_list.append(gts)
            result_list.append(Dis_output)
            result_list.append(Dis_target)
            visualize_list(result_list, option['log_path'])

            if rate == 1:
                loss_record.update(supervised_loss.data, option['batch_size'])
                supervised_loss_record.update(loss_all.data, option['batch_size'])
                dis_loss_record.update(dis_loss.data, option['batch_size'])

        progress_bar.set_postfix(loss=f'{loss_record.show():.3f}|{supervised_loss_record.show():.3f}|{dis_loss_record.show():.3f}')

    return {'generator': generator, "discriminator": discriminator}, loss_record
