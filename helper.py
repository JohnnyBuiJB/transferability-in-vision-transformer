import sys
import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append(os.path.abspath('references/TransUNet'))
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from datasets.dataset_synapse import Synapse_dataset
from networks.vit_seg_modeling import VisionTransformer as ViT_seg

# Set device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show_compare_img(original, adversarial, noise, ori_pred, adv_pred, ground_truth):
    gs = gridspec.GridSpec(4, 6)

    ax1 = plt.subplot(gs[:2, :2])
    ax1.set_title('Original')
    ax1.imshow(original, cmap='gray')

    ax2 = plt.subplot(gs[:2, 2:4])
    ax2.set_title('Adversarial')
    ax2.imshow(adversarial, cmap='gray')

    ax3 = plt.subplot(gs[:2, 4:])
    ax3.set_title('Noise')
    ax3.imshow(noise, cmap='gray')

    ax4 = plt.subplot(gs[2:, :2])
    ax4.set_title('Original Prediction')
    ax4.imshow(ori_pred, cmap='jet')

    ax5 = plt.subplot(gs[2:, 2:4])
    ax5.set_title('Adversarial Prediction')
    ax5.imshow(adv_pred, cmap='jet')

    ax6 = plt.subplot(gs[2:, 4:])
    ax6.set_title('Ground Trurth')
    ax6.imshow(ground_truth, cmap='jet')
    
    plt.tight_layout()
    plt.savefig('comparison.png')

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_TransUNet_model(args):
    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': 9,
            'z_spacing': 1,
            'img_size': 224,

        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.img_size = dataset_config[dataset_name]['img_size']
    args.is_pretrain = True

    snapshot_path = args.checkpoint_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    net.load_state_dict(torch.load(snapshot_path))
    return net