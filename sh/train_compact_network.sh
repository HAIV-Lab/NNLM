DSTORE_SIZE=3608731
DATA_PATH=/data/zqh/adaptive-knn-mt/store/data/multi-domain/it/data-bin
PROJECT_PATH=/data/zqh/adaptive-knn-mt
MODEL_PATH=/data/zqh/adaptive-knn-mt/store/models/wmt19.de-en.ffn8192.pt
DATASTORE_PATH=/data/zqh/adaptive-knn-mt/store/datastore/it_finetune

MODEL_RECORD_PATH=/data/zqh/adaptive-knn-mt/store/ablation_exp/both_gt_and_knn/it
CUDA_VISIBLE_DEVICES=0 python \
  $PROJECT_PATH/fairseq_cli/train_cmp.py \
  $DATA_PATH \
  --log-interval 100 --log-format simple \
  --arch transformer_wmt19_de_en_with_datastore \
  --save-dir "$MODEL_RECORD_PATH" --restore-file "$MODEL_PATH" \
  --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer \
  --validate-interval-updates 2000 --save-interval-updates 2000 --keep-interval-updates 1 --max-update 500000 --validate-after-updates 2000 \
  --save-interval 2000 --validate-interval 2000 \
  --keep-best-checkpoints 1 --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
  --train-subset train --valid-subset valid --source-lang de --target-lang en \
  --criterion cross_entropy \
  --max-source-positions 1024 --max-target-positions 1024 \
  --batch-size 4 --update-freq 16 --batch-size-valid 4 \
  --task translation \
  --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 --min-lr 3e-09 --lr 0.0003 --clip-norm 1.0 \
  --lr-scheduler reduce_lr_on_plateau --lr-patience 1 --lr-shrink 0.5 \
  --patience 30 --max-epoch 35 \
  --load-knn-datastore --dstore-filename $DATASTORE_PATH --use-knn-datastore \
  --dstore-fp16 --dstore-size $DSTORE_SIZE --probe 32 \
  --knn-sim-func do_not_recomp_l2 \
  --use-gpu-to-search --move-dstore-to-mem \
  --knn-lambda-type fix --knn-lambda-value 0.5 --knn-temperature-type fix --knn-temperature-value 10 --only-train-knn-parameter \
  --knn-k-type fix --k-lambda-net-hid-size 32 --k-lambda-net-dropout-rate 0.0 --max-k 4 --k 4 \
  --label-count-as-feature

    

  # --arch transformer_wmt19_de_en_with_datastore --optimizer adam \
  # --adam-betas '(0.9, 0.98)' \
  # --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --dropout 0.3 --warmup-init-lr 1e-07 --share-decoder-input-output-embed \
  # --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --encoder-ffn-embed-dim=8192 \
  # --max-tokens 2048 --batch-size 2 \
  # --keep-best-checkpoints 50 --patience 30 --ddp-backend no_c10d \
  # --seed 2022 --max-epoch 10  --warmup-updates 4000  \
  # --update-freq 16  --skip-invalid-size-inputs-valid-test \
  # --finetune-from-model $MODEL_PATH \
  # --maximize-best-checkpoint-metric --no-progress-bar --log-interval 20 --save-dir $MODEL_RECORD_PATH \
  # --keep-interval-updates 20 \
  # --load-knn-datastore --dstore-filename $DATASTORE_PATH --use-knn-datastore \
  # --dstore-fp16 --dstore-size $DSTORE_SIZE --probe 32 \
  # --knn-sim-func do_not_recomp_l2 \
  # --use-gpu-to-search --move-dstore-to-mem --no-load-keys \
  # --knn-lambda-type fix --knn-lambda-value 0.5 --knn-temperature-type fix --knn-temperature-value 10 --only-train-knn-parameter \
  # --knn-k-type fix --k-lambda-net-hid-size 32 --k-lambda-net-dropout-rate 0.0 --max-k 32 --k 32 \
  # --label-count-as-feature




