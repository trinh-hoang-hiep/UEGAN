
import torch

from transformers import SegformerModel
model = SegformerModel.from_pretrained("nvidia/mit-b5")
def get_backbone(option):
    if option['backbone'].lower() == 'swin':
        backbone=model.encoder

        channel_list = [64, 128, 320, 512]
    elif option['backbone'].lower() == 'r50':
        from model.backbone.resnet import ResNet50Backbone
        backbone = ResNet50Backbone()
        channel_list = [256, 512, 1024, 2048]
    elif option['backbone'].lower() == 'dpt':
        from model.backbone.DPT import DPT
        backbone = DPT().cuda()
        channel_list = [256, 512, 768, 768]

    return backbone, channel_list