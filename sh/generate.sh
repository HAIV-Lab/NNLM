data_dir=./nmt/data/multi-domain/subtitles

# data_dir=./nmt/general
# data_dir=./nmt/data/mix-domain/medical

CUDA_VISIBLE_DEVICES=0 python ./fairseq/fairseq_cli/generate.py ${data_dir}/data-bin \
--path "./nmt/models/wmt19.de-en.ffn8192.pt" \
--in-domain-model-checkpoint  "./checkpoints/checkpoints_general_big/checkpoint_best.pt" \
--skip-invalid-size-inputs-valid-test --beam 5 --remove-bpe > temp.txt
