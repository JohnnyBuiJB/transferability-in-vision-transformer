#!/bin/bash
onid=$(whoami)
# python attack.py --volume_path /nfs/hpc/share/$onid/data/Synapse/test_vol_h5 --checkpoint_path model/TU_Synapse224_1314/epoch_149.pth \
#     --list_dir references/TransUNet/lists/lists_Synapse --save_dir /nfs/hpc/share/$onid/data/Synapse/TransUNet_1314_adv_vol_h5 --dataset Synapse \
#     --cfg references/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml

python attack.py --volume_path /nfs/hpc/share/$onid/data/Synapse/test_vol_h5 --checkpoint_path /nfs/hpc/share/yonge/model/TU_Synapse224_1314/epoch_149.pth \
    --list_dir references/TransUNet/lists/lists_Synapse --save_dir /nfs/hpc/share/$onid/data/Synapse/TransUNet_1314_adv_vol_h5 --dataset Synapse \
    --vit_name R50-ViT-B_16 --model_name TU