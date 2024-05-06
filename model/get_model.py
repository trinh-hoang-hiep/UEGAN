import torch
from model.saliency_detector import discriminator, ebm_prior
from model.saliency_detector import sod_model, sod_model_with_vae
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

def get_model(option):

    if option['uncer_method'].lower() == 'ganabp': 
        model = sod_model(option=option).to(device)
        uncertainty_model = discriminator(ndf=64).to(device)
    else:
        raise NotImplementedError

    param_count = sum(x.numel()/1e6 for x in model.parameters()) 
    print("[INFO]: Model based on [{}] have {:.4f}Mb paramerters in total".format(option['model_name'], param_count)) 

    if option['checkpoint'] is not None: 

        print('Load checkpoint from {}'.format(option['checkpoint']))

    return model, uncertainty_model 
