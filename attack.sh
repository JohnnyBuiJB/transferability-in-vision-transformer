#!/bin/bash
onid=$(whoami)
craft_model=U_Synapse224_1314
save_dir=${craft_model}_adv_vol_h5
model_name=U

# python attack.py --volume_path /nfs/hpc/share/$onid/data/Synapse/test_vol_h5 --checkpoint_path /nfs/hpc/share/$onid/model/$craft_model/epoch_149.pth \
#     --list_dir references/TransUNet/lists/lists_Synapse --save_dir /nfs/hpc/share/$onid/data/Synapse/$save_dir --dataset Synapse \
#     --cfg references/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --model_name $model_name

python attack.py --volume_path /nfs/hpc/share/$onid/data/Synapse/test_vol_h5 --checkpoint_path /nfs/hpc/share/$onid/model/$craft_model/epoch_149.pth \
    --list_dir references/TransUNet/lists/lists_Synapse --save_dir /nfs/hpc/share/$onid/data/Synapse/$save_dir --dataset Synapse \
    --vit_name R50-ViT-B_16 --model_name U