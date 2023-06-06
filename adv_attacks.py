import sys
import os

import torch
import numpy as np
from scipy.ndimage import zoom

sys.path.append(os.path.abspath('references/TransUNet'))
sys.path.append(os.path.abspath('.'))

from torch.nn.modules.loss import CrossEntropyLoss
from utils import DiceLoss

from helper import device

def FGSM(ori_img, input, label, model, epsilon=0.01):
    input, label = input.to(device), label.to(device)
    input.requires_grad = True

    output = model(input)

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(output.shape[1])
    loss_ce = ce_loss(output, label[:].long())
    loss_dice = dice_loss(output, label, softmax=True)
    loss = 0.5 * loss_ce + 0.5 * loss_dice

    model.zero_grad()
    loss.backward()
    input_grad_sign = input.grad.data.squeeze().sign().cpu().detach().numpy()
    image_grad_sign = zoom(input_grad_sign, (ori_img.shape[0]/input.shape[2], ori_img.shape[1]/input.shape[3]), order=3)
    perturb_img = ori_img + epsilon*image_grad_sign
    perturb_img = torch.clip(torch.from_numpy(perturb_img), 0, 1)

    return perturb_img