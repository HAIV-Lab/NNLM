#!/bin/bash 

# the pre-trained general-domain checkpoint
ckt=./checkpoints/checkpoints_general_big/checkpoint_best.pt

# path to save the pruned checkpoint
save_ckt=./checkpoints/prune/prune_checkpoint.pt

# path to save the mask matrix 
save_mask=./mask/mask_matrix.txt

# prune ratio
ratio=0.3

python magnitude.py --pre-ckt-path $ckt --save-ckt-path $save_ckt \
						--save-mask-path $save_mask --prune-ratio $ratio
