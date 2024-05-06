import copy
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as sn
from utils import torch_tile, reparametrize
from model.backbone.get_backbone import get_backbone
from model.neck.get_neck import get_neck
from model.decoder.get_decoder import get_decoder
from model.depth_module.get_depth_module import get_depth_module

from model.blocks.base_blocks import FeatureFusionBlock


class sod_model(torch.nn.Module): 
    def __init__(self, option): 
        super(sod_model, self).__init__() 

        self.backbone, self.channel_list = get_backbone(option) 
        self.neck = get_neck(option, self.channel_list) 
        self.decoder = get_decoder(option) 
        self.depth_module = get_depth_module(option, self.channel_list) 
        self.noise_model = noise_model(option)   

    def forward(self, img, z=None, gts=None, depth=None, coord=None, cell=None):
        if depth is not None: 
            if 'head' in self.depth_module.keys():
                img = self.depth_module['head'](img, depth)
            elif 'feature' in self.depth_module.keys():
                depth_features = self.depth_module['feature'](depth)
            elif 'rgb' in self.depth_module.keys():
                img = img

        
        backbone_features = self.backbone(img,output_hidden_states=True).hidden_states
        
        
        
        neck_features = self.neck(backbone_features) 

        if z is not None: 
            neck_features = self.noise_model(z, neck_features) 

        if depth is not None and 'fusion' in self.depth_module.keys(): 
            neck_features = self.depth_module['fusion'](neck_features, depth_features)

        
        
        outputs = self.decoder(neck_features, coord=coord, cell=cell) 
        if depth is not None and 'aux_decoder' in self.depth_module.keys(): 
            outputs_depth = self.depth_module['aux_decoder'](backbone_features)
            return {'sal_pre': outputs, 'depth_pre': outputs_depth, 'backbone_features':backbone_features}
        else:
            return {'sal_pre': outputs, 'depth_pre': None, 'backbone_features':backbone_features} 


class sod_model_with_vae(torch.nn.Module):
    def __init__(self, option):
        super(sod_model_with_vae, self).__init__()

        self.backbone, self.channel_list = get_backbone(option)
        self.neck_prior = get_neck(option, self.channel_list)
        self.decoder_prior = get_decoder(option)
        self.depth_module = get_depth_module(option, self.channel_list)
        self.vae_model = vae_model(option)
        self.decoder_post = copy.deepcopy(self.decoder_prior)
        self.neck_post = copy.deepcopy(self.neck_prior)

    def forward(self, img, z=None, gts=None, depth=None):
        if depth is not None:
            if 'head' in self.depth_module.keys():
                img = self.depth_module['head'](img, depth)
            elif 'feature' in self.depth_module.keys():
                depth_features = self.depth_module['feature'](depth)
        
        backbone_features = self.backbone(img)
        neck_features_prior = self.neck_prior(backbone_features)
        neck_features_post = self.neck_post(backbone_features)
        vae_model_input = [img, neck_features_prior, neck_features_post, gts]
        neck_features_z_prior, neck_features_z_post, kld = self.vae_model(*vae_model_input)
        
        

        if gts is not None:   
            outputs_prior = self.decoder_prior(neck_features_z_prior)
            outputs_post = self.decoder_post(neck_features_z_post)

            return outputs_prior, outputs_post, kld
        else:   
            outputs = self.decoder_prior(neck_features_z_prior)
            return {'sal_pre': outputs, 'depth_pre': None, 'backbone_features':backbone_features}


class vae_model(nn.Module):
    def __init__(self, option):
        super(vae_model, self).__init__()
        self.enc_x = encode_for_vae(input_channels=3, option=option)
        self.enc_xy = encode_for_vae(input_channels=4, option=option)

        self.noise_model_prior = noise_model(option)
        self.noise_model_post = noise_model(option)

    def forward(self, img, neck_features_prior, neck_features_post, y=None):
        if y is None:
            mu_prior, logvar_prior, _ = self.enc_x(img)
            z_prior = reparametrize(mu_prior, logvar_prior)
            neck_features_z_prior = self.noise_model_prior(z_prior, neck_features_prior)

            return neck_features_z_prior, None, None
        else:
            mu_prior, logvar_prior, dist_prior = self.enc_x(img)
            z_prior = reparametrize(mu_prior, logvar_prior)
            neck_features_z_prior = self.noise_model_prior(z_prior, neck_features_prior)

            mu_post, logvar_post, dist_post = self.enc_xy(torch.cat((img, y), 1))
            z_post = reparametrize(mu_post, logvar_post)
            neck_features_z_post = self.noise_model_post(z_post, neck_features_post)

            kld = torch.mean(torch.distributions.kl.kl_divergence(dist_post, dist_prior))
            return neck_features_z_prior, neck_features_z_post, kld


class noise_model(nn.Module):
    def __init__(self, option):
        super(noise_model, self).__init__() 
        in_channel = option['neck_channel'] + option['latent_dim'] 
        out_channel = option['neck_channel'] 
        self.noise_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0) 

    def process_z_noise(self, z, feat):
        spatial_axes = [2, 3] 
        z_noise = torch.unsqueeze(z, 2) 
        z_noise = torch_tile(z_noise, 2, feat.shape[spatial_axes[0]]) 
        z_noise = torch.unsqueeze(z_noise, 3) 
        z_noise = torch_tile(z_noise, 3, feat.shape[spatial_axes[1]]) 

        return z_noise 

    def forward(self, z, neck_features):
        z_noise = self.process_z_noise(z, neck_features[-1]) 
        neck_feat_with_noise = self.noise_conv(torch.cat((neck_features[-1], z_noise), 1)) 
        neck_features[-1] = neck_feat_with_noise 
        return neck_features 


class discriminator(nn.Module):
    def __init__(self, ndf):
        super(discriminator, self).__init__() 
        self.conv1 = nn.Conv2d(4, ndf, kernel_size=3, stride=2, padding=1) 
        self.conv2 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1) 
        self.conv3 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1) 
        self.conv4 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1) 
        self.classifier = nn.Conv2d(ndf, 1, kernel_size=3, stride=2, padding=1) 
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True) 
        self.bn1 = nn.BatchNorm2d(ndf) 
        self.bn2 = nn.BatchNorm2d(ndf) 
        self.bn3 = nn.BatchNorm2d(ndf) 
        self.bn4 = nn.BatchNorm2d(ndf) 

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.leaky_relu(self.bn4(self.conv4(x)))

        x = self.classifier(x)
        return x


class encode_for_vae(nn.Module):
    def __init__(self, input_channels, option):
        super(encode_for_vae, self).__init__()
        channels = option['neck_channel']
        latent_size = option['latent_dim']
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(channels, 2*channels, kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(2*channels, 4*channels, kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(4*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(8*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels*2)
        self.bn3 = nn.BatchNorm2d(channels*4)
        self.bn4 = nn.BatchNorm2d(channels*8)
        self.bn5 = nn.BatchNorm2d(channels*8)
        self.channel = channels
        self.hidden_size = option['trainsize'] // 32

        self.fc1 = nn.Linear(channels*8*self.hidden_size*self.hidden_size, latent_size)  
        self.fc2 = nn.Linear(channels*8*self.hidden_size*self.hidden_size, latent_size)  

        self.leakyrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.leakyrelu(self.bn1(self.layer1(input)))
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        output = self.leakyrelu(self.bn5(self.layer5(output)))
        output = output.view(-1, self.channel*8*self.hidden_size*self.hidden_size)  
        

        mu = self.fc1(output)
        logvar = self.fc2(output)
        dist = torch.distributions.Independent(torch.distributions.Normal(loc=mu, scale=torch.exp(logvar)), 1)

        return mu, logvar, dist


class ebm_prior(nn.Module):
    def __init__(self, ebm_out_dim, ebm_middle_dim, latent_dim):
        super().__init__()
        e_sn = False
        apply_sn = sn if e_sn else lambda x: x

        self.ebm = nn.Sequential(
            apply_sn(nn.Linear(latent_dim, ebm_middle_dim)),
            torch.nn.GELU(),
            apply_sn(nn.Linear(ebm_middle_dim, ebm_middle_dim)),
            torch.nn.GELU(),
            apply_sn(nn.Linear(ebm_middle_dim, ebm_out_dim))
        )
        self.ebm_out_dim = ebm_out_dim

    def forward(self, z):
        return self.ebm(z).view(-1, self.ebm_out_dim, 1, 1)





































































































































































































