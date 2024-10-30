# save dir
save=kd_koran_checkpoints

# the pruned checkpoint
ckt=./checkpoints/prune/prune_checkpoint.pt

# the general domain checkpoint
teacher_ckt=./checkpoints/checkpoints_general_big/checkpoint_best.pt

# the absolute path to the mask file
mask=./mask/mask_matrix_big.txt

data_dir=./nmt/data/multi-domain/koran

python  ./fairseq/train.py ${data_dir}/data-bin \
    --arch transformer_vaswani_wmt_en_de_big  --fp16 --reset-optimizer \
	    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
		--mask-file $mask --knowledge-distillation True --checkpoint_kd True --teacher_model transformer_vaswani_wmt_en_de_big \
         --lr-scheduler fixed --restore-file $ckt --restore-teacher-file $teacher_ckt \
	        --lr 7.5e-5 --dropout 0.1 --ddp-backend no_c10d \
           --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
		    --max-tokens  4096  --save-dir checkpoints/$save  --save-interval 1 \
		    --update-freq 16 --no-progress-bar --log-format json --log-interval 25 \
			--share-decoder-input-output-embed --encoder-ffn-embed-dim=8192 \
			--seed 2022 --max-epoch 50  --warmup-updates 4000 \


          
