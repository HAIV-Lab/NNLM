DSTORE_SIZE=3608731
DATA_PATH=/data/zqh/adaptive-knn-mt/store/data/multi-domain/it/data-bin
PROJECT_PATH=/data/zqh/adaptive-knn-mt
MODEL_PATH=/data/zqh/adaptive-knn-mt/store/ablation_exp/both_gt_and_knn/it/checkpoint.best_loss_1.87.pt
DATASTORE_PATH=/data/zqh/adaptive-knn-mt/store/datastore/it_finetune

# max_k_grid=(4 8 16 32)
max_k_grid=(4)
# batch_size_grid=(32 32 32 32)
batch_size_grid=(8)
# update_freq_grid=(1 1 1 1)
update_freq_grid=(1)
# valid_batch_size_grid=(32 32 32 32)
valid_batch_size_grid=(8)

for idx in ${!max_k_grid[*]}
do

  MODEL_RECORD_PATH=/data/zqh/adaptive-knn-mt/store/ablation_exp/use_adaptive/it
#   TRAINING_RECORD_PATH=/path/to/save/tensorboard/train-hid32-maxk${max_k_grid[$idx]}
#   mkdir -p "$TRAINING_RECORD_PATH"

  CUDA_VISIBLE_DEVICES=0 python \
  $PROJECT_PATH/fairseq_cli/train.py \
  $DATA_PATH \
  --log-interval 100 --log-format simple \
  --arch transformer_wmt19_de_en_with_datastore \
  --save-dir "$MODEL_RECORD_PATH" --best-checkpoint-metric loss --restore-file "$MODEL_PATH" \
  --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer \
  --validate-interval-updates 200 --save-interval-updates 200 --keep-interval-updates 1 --max-update 500000 --validate-after-updates 200 \
  --save-interval 100 --validate-interval 100 \
  --keep-best-checkpoints 1 --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
  --train-subset valid --valid-subset valid --source-lang de --target-lang en \
  --criterion cross_entropy \
  --max-source-positions 1024 --max-target-positions 1024 \
  --batch-size "${batch_size_grid[$idx]}" --update-freq "${update_freq_grid[$idx]}" --batch-size-valid "${valid_batch_size_grid[$idx]}" \
  --task translation \
  --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 --min-lr 3e-05 --lr 0.0003 --clip-norm 1.0 \
  --lr-scheduler reduce_lr_on_plateau --lr-patience 5 --lr-shrink 0.5 \
  --patience 30 --max-epoch 30 \
  --load-knn-datastore --dstore-filename $DATASTORE_PATH --use-knn-datastore \
  --dstore-fp16 --dstore-size $DSTORE_SIZE --probe 32 \
  --knn-sim-func do_not_recomp_l2 \
  --use-gpu-to-search --move-dstore-to-mem \
  --knn-lambda-type trainable --knn-temperature-type fix --knn-temperature-value 10 --only-train-knn-parameter \
  --knn-k-type fix --k-lambda-net-hid-size 32 --k-lambda-net-dropout-rate 0.0 --max-k "${max_k_grid[$idx]}" --k "${max_k_grid[$idx]}" 
done