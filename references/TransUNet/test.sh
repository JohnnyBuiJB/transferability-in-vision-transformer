#!/bin/bash
onid=$(whoami)
test_model="TU_Synapse224_1314"
adv_model="SU_Synapse224"
python test.py --volume_path /nfs/hpc/share/$onid/data/Synapse/${adv_model}_adv_vol_h5 \
               --checkpoint_path /nfs/hpc/share/$onid/model/$test_model/epoch_149.pth \
               --is_savenii --vit_name R50-ViT-B_16 --test_save_dir /nfs/hpc/share/$onid/predictions/$test_model/${adv_model}_adv