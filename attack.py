import argparse
import sys
import os
import random
import logging
import getpass as gt

import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader
from scipy.ndimage import zoom
from tqdm import tqdm

from helper import set_seeds, get_TransUNet_model, show_compare_img, get_SwinUnet_model
from adv_attacks import FGSM
from utils import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--checkpoint_path', type=str,
                    default='references/model/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth', help='path points to model weight checkpoint') 
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--save_dir', type=str,
                    default=f'/nfs/hpc/share/{gt.getuser()}/data/Synapse/adv_vol_h5', help='specify the directory to store adversarial examples')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--vit_name', type=str, default='ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--model_name', type=str, default='TU', help='select between (TU, SU, BU or U)')
parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file')

args = parser.parse_args()

def craft_single_volume_attack(image, label, net, patch_size=[256, 256], save_path=None, case=None, attack=FGSM):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    adv_volume = np.zeros_like(image)
    for ind in range(image.shape[0]):
        image_slice = image[ind, :, :]
        label_slice = label[ind, :, :]
        x, y = image_slice.shape[0], image_slice.shape[1]

        # Zoom in image
        if x != patch_size[0] or y != patch_size[1]:
            image_slice = zoom(image_slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            label_slice = zoom(label_slice, (patch_size[0] / x, patch_size[1] / y), order=0)

        image_slice = torch.from_numpy(image_slice).unsqueeze(0).unsqueeze(0).float().cuda()
        label_slice = torch.from_numpy(label_slice).unsqueeze(0).float().cuda()
        
        # Craft adversarial slice
        adv_slice = attack(image[ind, :, :], image_slice, label_slice, net).squeeze().cpu().detach()

        # Zoom out image
        # if x != patch_size[0] or y != patch_size[1]:
        #     adv_slice = zoom(adv_slice, (x / patch_size[0], y / patch_size[1]), order=3)
        
        # adv_slice = normalize_image(adv_slice)
        adv_volume[ind] = adv_slice

    if save_path:
        hf = h5py.File(f'{save_path}/{case}.npy.h5', 'w')
        hf.create_dataset('image', data=adv_volume)
        hf.create_dataset('label', data=label)

    return adv_volume


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.info(str(args))

    save_path = args.save_dir
    os.makedirs(save_path, exist_ok=True)
    
    if args.model_name == 'TU':
        model = get_TransUNet_model(args)
    elif args.model_name == 'SU':
        model = get_SwinUnet_model(args)

    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        print(f"Crafting {case_name}...")
        adv_image = craft_single_volume_attack(image, label, model, patch_size=[args.img_size, args.img_size],
                                      save_path=save_path, case=case_name, attack=FGSM)
        # adv_image = torch.from_numpy(adv_image).unsqueeze(0)
        # print("Predicting...")
        # metric_i = test_single_volume(adv_image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
        #                         test_save_path='./references/predictions/TU_Synapse224/adversarial', case=case_name)
        # logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))

    # loader = iter(testloader)
    # sample = next(loader)
    # sample = next(loader)
    # image, label = sample['image'], sample['label']
    # image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0)

    # # Get random image to test on adversarial attack
    # idx = random.randint(0, image.shape[0])
    # print(idx)
    # image_slice = image[idx, :, :]
    # label_slice = label[idx, :, :]
    # x, y = image_slice.shape[0], image_slice.shape[1]

    # # Zoom in image to get correct image size (224,224)
    # if x != args.img_size or y != args.img_size:
    #     image_slice = zoom(image_slice, (args.img_size / x, args.img_size / y), order=3)  # previous using 0
    #     label_slice = zoom(label_slice, (args.img_size / x, args.img_size / y), order=0)

    # ori_image = torch.from_numpy(image_slice).unsqueeze(0).unsqueeze(0).float().cuda() # (1, 1, 224, 224)
    # label = torch.from_numpy(label_slice).unsqueeze(0).float().cuda() # (1, 224, 224)

    # adv_image = FGSM(ori_image, label, model)
    # adv_image = normalize_image(adv_image)

    # model.eval()
    # clean_pred = model(ori_image)
    # clean_pred = torch.argmax(torch.softmax(clean_pred, dim=1), dim=1).squeeze(0)
    # ori_image = zoom(ori_image.squeeze().cpu().detach(), (x / args.img_size, y / args.img_size), order=0)
    # ori_image = normalize_image(ori_image)
    # clean_pred = zoom(clean_pred.cpu().detach(),  (x / args.img_size, y / args.img_size), order=0)

    # adv_pred = model(adv_image)
    # adv_pred = torch.argmax(torch.softmax(adv_pred, dim=1), dim=1).squeeze(0)
    # adv_image = zoom(adv_image.squeeze().cpu().detach(),  (x / args.img_size, y / args.img_size), order=0)   
    # adv_pred = zoom(adv_pred.cpu().detach(),  (x / args.img_size, y / args.img_size), order=0)

    # # Move to cpu for plotting
    # ori_image = ori_image.squeeze()
    # adv_image = adv_image.squeeze()
    # noise =  adv_image - ori_image
    # clean_pred = clean_pred
    # adv_pred = adv_pred
    # label = label.squeeze().cpu()

    # show_compare_img(ori_image, adv_image, noise, clean_pred, adv_pred, label)