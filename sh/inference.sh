DSTORE_SIZE=3608731
DATA_PATH=/data/zqh/adaptive-knn-mt/store/data/multi-domain/it/data-bin
PROJECT_PATH=/data/zqh/adaptive-knn-mt
MODEL_PATH=/data/zqh/adaptive-knn-mt/store/ablation_exp/both_gt_and_knn/it/checkpoint.best_loss_0.04.pt

DATASTORE_PATH=/data/zqh/adaptive-knn-mt/store/datastore/it_finetune

OUTPUT_PATH=/data/zqh/adaptive-knn-mt/store/result/it

mkdir -p "$OUTPUT_PATH"

CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/experimental_generate.py $DATA_PATH \
    --gen-subset test \
    --path "$MODEL_PATH" --arch transformer_wmt19_de_en_with_datastore \
    --beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
    --scoring sacrebleu \
    --batch-size 8 \
    --tokenizer moses --remove-bpe \
    --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True,
    'dstore_filename': '$DATASTORE_PATH', 'dstore_size': $DSTORE_SIZE, 'dstore_fp16': True, 'probe': 32,
    'knn_sim_func': 'do_not_recomp_l2', 'use_gpu_to_search': True, 'move_dstore_to_mem': True, 'no_load_keys': True,
    'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_lambda_value': 0.6, 'knn_lambda_type': 'fix', 'knn_k_type': 'fix',
    'knn_record_index': True }" \
    | tee "$OUTPUT_PATH"/generate.txt

grep ^S "$OUTPUT_PATH"/generate.txt | cut -f2- > "$OUTPUT_PATH"/src
grep ^T "$OUTPUT_PATH"/generate.txt | cut -f2- > "$OUTPUT_PATH"/ref
grep ^H "$OUTPUT_PATH"/generate.txt | cut -f3- > "$OUTPUT_PATH"/hyp
grep ^D "$OUTPUT_PATH"/generate.txt | cut -f3- > "$OUTPUT_PATH"/hyp.detok

# 41.42
# 32.57 finetune 的模型 训练 compact 网络 IT 领域， 计算 compact 和 knn 分布 ce 损失， target 和 lprobs ce 损失
