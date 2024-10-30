DSTORE_SIZE=19070000
MODEL_PATH=/data/zqh/NLP/adaptive-knn-mt/store/checkpoints/medical/checkpoint_best.pt
DATA_PATH=/data/zqh/NLP/adaptive-knn-mt/store/data/multi-domain/medical/data-bin
DATASTORE_PATH=/data/zqh/NLP/adaptive-knn-mt/store/datastore/medical_finetune
PROJECT_PATH=/data/zqh/NLP/adaptive-knn-mt
# it 3608731 
# koran 523700
# law 19070000
# medical 6903320


mkdir -p $DATASTORE_PATH

CUDA_VISIBLE_DEVICES=1 python $PROJECT_PATH/save_datastore.py $DATA_PATH \
    --dataset-impl mmap \
    --task translation \
    --valid-subset train \
    --path $MODEL_PATH \
    --max-tokens 4096 --skip-invalid-size-inputs-valid-test \
    --decoder-embed-dim 1024 --dstore-fp16 --dstore-size $DSTORE_SIZE --dstore-mmap $DATASTORE_PATH
