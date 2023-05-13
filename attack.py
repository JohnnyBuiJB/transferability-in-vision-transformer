import argparse
import sys
import os
import random

import torch
from torch.utils.data import DataLoader
from scipy.ndimage import zoom

from helper import set_seeds, get_TransUNet_model, show_compare_img
from adv_attacks import FGSM

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--checkpoint_path', type=str,
                    default='references/model/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth', help='path points to model weight checkpoint')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--vit_name', type=str, default='ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--seed', type=int, default=1234, help='random seed')

args = parser.parse_args()

if __name__ == '__main__':
    # set_seeds(args.seed)
    model = get_TransUNet_model(args)

    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    loader = iter(testloader)
    sample = next(loader)
    sample = next(loader)
    image, label = sample['image'], sample['label']
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0)

    # Get random image to test on adversarial attack
    idx = random.randint(0, image.shape[0])
    print(idx)
    image_slice = image[idx, :, :]
    label_slice = label[idx, :, :]
    x, y = image_slice.shape[0], image_slice.shape[1]

    # Zoom in image to get correct image size (224,224)
    if x != args.img_size or y != args.img_size:
        image_slice = zoom(image_slice, (args.img_size / x, args.img_size / y), order=3)  # previous using 0
        label_slice = zoom(label_slice, (args.img_size / x, args.img_size / y), order=0)

    ori_image = torch.from_numpy(image_slice).unsqueeze(0).unsqueeze(0).float().cuda() # (1, 1, 224, 224)
    label = torch.from_numpy(label_slice).unsqueeze(0).float().cuda() # (1, 224, 224)

    adv_image = FGSM(ori_image, label, model)

    model.eval()
    clean_pred = model(ori_image)
    clean_pred = torch.argmax(torch.softmax(clean_pred, dim=1), dim=1).squeeze(0)

    adv_pred = model(adv_image)
    adv_pred = torch.argmax(torch.softmax(adv_pred, dim=1), dim=1).squeeze(0)

    # Move to cpu for plotting
    ori_image = ori_image.squeeze().cpu().detach()
    adv_image = adv_image.squeeze().cpu().detach()
    noise = adv_image - ori_image
    clean_pred = clean_pred.cpu().detach()
    adv_pred = adv_pred.cpu().detach()
    label = label.squeeze().cpu().detach()

    show_compare_img(ori_image, adv_image, noise, clean_pred, adv_pred, label)