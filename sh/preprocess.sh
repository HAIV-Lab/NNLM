data_dir=./store/data/multi-domain/it
model_dir=./store/models
fairseq-preprocess --source-lang de --target-lang en \
    --srcdict ${model_dir}/dict.de.txt --tgtdict ${model_dir}/dict.en.txt \
    --trainpref ${data_dir}/train --validpref ${data_dir}/valid --testpref ${data_dir}/test \
    --destdir ${data_dir}/data-bin \

