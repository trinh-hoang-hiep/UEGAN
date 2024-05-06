import torch
import numpy as np
from loss.structure_loss import structure_loss



def bce_loss_with_sigmoid(pred, gt, weight=None):
    return torch.nn.functional.binary_cross_entropy_with_logits(pred, gt, reduce='none')


def get_loss(option): #1
    if option['loss'] == 'structure': #1
        loss_fun = structure_loss #1
    elif option['loss'] == 'bce':
        loss_fun = bce_loss_with_sigmoid
    elif option['loss'] == 'weak':
        loss_fun = weakly_loss(option)

    return loss_fun #1


def cal_loss(pred, gt, loss_fun, weight=None):
    if isinstance(pred, list):
        loss_sum = 0
        for i in pred:
            loss_curr = loss_fun(i, gt, weight)
            loss_sum += loss_curr
        loss = loss_sum / len(pred)
    elif isinstance(pred, dict):
        import pdb; pdb.set_trace()
        loss = 0
        for key in pred.keys():
            loss_curr = loss_fun(pred(key), gt, weight)
            loss += loss_curr
        loss = loss / len(pred)
    else:
        import pdb; pdb.set_trace()
        loss = loss_fun(pred, gt, weight)

    return loss
