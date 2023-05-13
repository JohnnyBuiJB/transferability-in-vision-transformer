import sys
import os

import torch
from scipy.ndimage import zoom

sys.path.append(os.path.abspath('references/TransUNet'))

from torch.nn.modules.loss import CrossEntropyLoss
from utils import DiceLoss

from helper import device

def FGSM(image, label, model, epsilon=0.01):
    image, label = image.to(device), label.to(device)
    image.requires_grad = True

    output = model(image)

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(output.shape[1])
    loss_ce = ce_loss(output, label[:].long())
    loss_dice = dice_loss(output, label, softmax=True)
    loss = 0.5 * loss_ce + 0.5 * loss_dice

    model.zero_grad()
    loss.backward()
    image_grad = image.grad.data
    perturb_img = image + epsilon*image_grad.sign()
    perturb_img = torch.clamp(perturb_img, 0, 1)

    return perturb_img