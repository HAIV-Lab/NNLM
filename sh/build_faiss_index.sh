PROJECT_PATH=/data/zqh/NLP/adaptive-knn-mt
DSTORE_PATH=/data/zqh/NLP/adaptive-knn-mt/store/datastore/it_finetune
DSTORE_SIZE=3608731

CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/train_datastore_gpu.py \
  --dstore_mmap $DSTORE_PATH \
  --dstore_size $DSTORE_SIZE \
  --faiss_index ${DSTORE_PATH}/knn_index \
  --dstore-fp16 \
  --ncentroids 4096 \
  --probe 32 \
  --dimension 1024