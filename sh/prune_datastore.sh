DSTORE_SIZE=3608731
MODEL_PATH=/data/zqh/NLP/adaptive-knn-mt/store/checkpoints/it/checkpoint_finetune.pt
DATA_PATH=/data/zqh/NLP/adaptive-knn-mt/store/data/multi-domain/it/data-bin
DATASTORE_PATH=/data/zqh/NLP/adaptive-knn-mt/store/datastore/it_finetune
PROJECT_PATH=/data/zqh/NLP/adaptive-knn-mt

CUDA_VISIBLE_DEVICES=1 python $PROJECT_PATH/prune_datastore/knn_prune.py $DATA_PATH \
    --dataset-impl mmap \
    --task translation \
    --valid-subset train \
    --path $MODEL_PATH \
    --max-tokens 4096 --skip-invalid-size-inputs-valid-test \
    --dstore-filename $DATASTORE_PATH \
    --decoder-embed-dim 1024 --dstore-fp16 --dstore-size $DSTORE_SIZE --probe 32 \
    --knn-sim-func do_not_recomp_l2 \
    --use-gpu-to-search --move-dstore-to-mem --no-load-keys \
    --k 4 