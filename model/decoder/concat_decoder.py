import torch  
import torch.nn as nn 
from torch.nn import functional as F 
from model.blocks.rcab_block import RCAB 
from model.blocks.base_blocks import SimpleHead 
from model.neck.neck_blocks import ASPP_Module 
import model.decoder.grid_sample_gradfix as grid_sample_gradfix 


if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
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

class concat_decoder(torch.nn.Module): 
    def __init__(self, option): 
        super(concat_decoder, self).__init__() 
        self.channel_size = option['neck_channel'] 
        self.deep_sup = option['deep_sup'] 
        self.fc = nn.Linear(2144,134) 
        self.rcab_conv = RCAB(134)
        self.aspp_head = ASPP_Module(dilation_series=[3, 6, 12, 18], padding_series=[3, 6, 12, 18], 
                                     out_channel=1, input_channel=134)

    def forward(self, features, coord=None, cell=None): 
        up_feat_list = [] 
        for i, feat in enumerate(features): 
            up_feat_list.append(nn.functional.interpolate(feat, scale_factor=(2**i), mode='bilinear', align_corners=True)) 

        feat = torch.cat(up_feat_list, dim=1) 
        
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        rx = 2 / feat.shape[-2] / 2 
        ry = 2 / feat.shape[-1] / 2 
        feat_coord = make_coord(feat.shape[-2:], flatten=False).to(device).permute(2, 0, 1).unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:]) 

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                
                q_feat = grid_sample_gradfix.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1))[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = grid_sample_gradfix.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1))[:, :, 0, :] \
                    .permute(0, 2, 1)

                rel_coord = coord - q_coord 
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord, coord], dim=-1)
                
                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([inp, rel_cell.to(device)], dim=-1)

                bs, q = coord.shape[:2] 
                
                inpp=nn.functional.interpolate(inp.view(bs,384,384,-1).permute(0, 3, 1,2), scale_factor=0.25, mode='bilinear', align_corners=True)
                pred = nn.functional.interpolate(self.aspp_head(self.rcab_conv(inpp)), scale_factor=4, mode='bilinear', align_corners=True)

                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0

        for pred, area in zip(preds, areas):
            ret = ret + pred.view(bs,147456,1) * (area / tot_area).unsqueeze(-1)
        ret=ret.view(bs,1,384,-1)
        return [ret] 

        

        
