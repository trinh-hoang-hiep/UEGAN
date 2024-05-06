import torch.nn as nn


def get_depth_module(option, in_channel_list):
    depth_module = nn.ModuleDict() #6
    if option['task'].lower() != 'rgbd-sod': #6
        depth_module = None #6
    else:
        if option['fusion'].lower() == 'early':
            from model.depth_module.early_fusion import early_fusion_conv
            depth_module['head'] = early_fusion_conv()
        elif option['fusion'].lower() == 'late':
            from model.depth_module.depth_feature import depth_feature
            from model.depth_module.feature_fusion import feature_fusion
            depth_module['feature'] = depth_feature(in_planes=128, out_planes=option['neck_channel'])
            depth_module['fusion'] = feature_fusion(option=option)
        elif option['fusion'].lower() == 'rgb':
            depth_module['rgb'] = nn.ModuleList()
        elif option['fusion'].lower() == 'aux':
            from model.decoder.concat_decoder import concat_decoder
            from model.neck.neck import basic_neck
            depth_module['aux_decoder'] = nn.Sequential(basic_neck(in_channel_list, option['neck_channel']),
                                                        concat_decoder(option=option))

    return depth_module #6
