DATA_PATH=/data/zqh/adaptive-knn-mt/store/data/multi-domain/law/data-bin
PROJECT_PATH=/data/zqh/adaptive-knn-mt
MODEL_PATH=/data/zqh/adaptive-knn-mt/store/models/wmt19.de-en.ffn8192.pt

# max_k_grid=(4 8 16 32)
max_k_grid=(16)
# batch_size_grid=(32 32 32 32)
batch_size_grid=(8)
# update_freq_grid=(1 1 1 1)
update_freq_grid=(1)
# valid_batch_size_grid=(32 32 32 32)
valid_batch_size_grid=(8)

MODEL_RECORD_PATH=/data/zqh/adaptive-knn-mt/store/checkpoints/it_finetune
CUDA_VISIBLE_DEVICES=0 python \
$PROJECT_PATH/fairseq_cli/train.py \
$DATA_PATH \
--arch transformer_wmt19_de_en_with_datastore --optimizer adam \
--adam-betas '(0.9, 0.98)' --fp16 \
--clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --dropout 0.3 --warmup-init-lr 1e-07 --share-decoder-input-output-embed \
--weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --encoder-ffn-embed-dim=8192 \
--max-tokens 2048 \
--keep-best-checkpoints 50 --patience 30 --ddp-backend no_c10d \
--seed 2022 --max-epoch 10  --warmup-updates 4000  \
--update-freq 8  --skip-invalid-size-inputs-valid-test \
--finetune-from-model $MODEL_PATH \
--maximize-best-checkpoint-metric --no-progress-bar --log-interval 20 --save-dir $MODEL_RECORD_PATH \
--keep-interval-updates 20 \
--knn-lambda-type fix --knn-k-type fix --knn-temperature-type fix 

