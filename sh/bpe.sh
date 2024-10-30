set -o errexit

model_dir=./store/models
# data_dir=./nmt/data/multi-domain/medical
data_dir=./store/data/multi-domain/law
BPEROOT=./fastBPE

${BPEROOT}/fast applybpe ${data_dir}/train.en ${data_dir}/train.tok.en  ${model_dir}/ende30k.fastbpe.code 
${BPEROOT}/fast applybpe ${data_dir}/train.de ${data_dir}/train.tok.de  ${model_dir}/ende30k.fastbpe.code 
${BPEROOT}/fast applybpe ${data_dir}/test.en ${data_dir}/test.tok.en  ${model_dir}/ende30k.fastbpe.code 
${BPEROOT}/fast applybpe ${data_dir}/test.de ${data_dir}/test.tok.de  ${model_dir}/ende30k.fastbpe.code 
${BPEROOT}/fast applybpe ${data_dir}/valid.en ${data_dir}/valid.tok.en  ${model_dir}/ende30k.fastbpe.code 
${BPEROOT}/fast applybpe ${data_dir}/valid.de ${data_dir}/valid.tok.de  ${model_dir}/ende30k.fastbpe.code 

# python ${BPEROOT}/learn_joint_bpe_and_vocab.py --input ${data_dir}/train.tok.en  -s 32000 -o ${data_dir}/bpecode.en --write-vocabulary ${data_dir}/voc.en
# python ${BPEROOT}/learn_joint_bpe_and_vocab.py --input ${data_dir}/train.tok.de  -s 32000 -o ${data_dir}/bpecode.de --write-vocabulary ${data_dir}/voc.de

# python ${BPEROOT}/apply_bpe.py -c ${model_dir}/ende30k.fastbpe.code  < ${data_dir}/norm.tok.en > ${data_dir}/norm.tok.bpe.en
# python ${BPEROOT}/apply_bpe.py -c ${model_dir}/ende30k.fastbpe.code  < ${data_dir}/norm.tok.en > ${data_dir}/norm.tok.bpe.en

# python ${BPEROOT}/apply_bpe.py -c ${model_dir}/ende30k.fastbpe.code  < ${data_dir}/train.tok.en > ${data_dir}/train.tok.bpe.en
# python ${BPEROOT}/apply_bpe.py -c ${model_dir}/ende30k.fastbpe.code  < ${data_dir}/valid.tok.en > ${data_dir}/valid.tok.bpe.en
# python ${BPEROOT}/apply_bpe.py -c ${model_dir}/ende30k.fastbpe.code  < ${data_dir}/test.tok.en > ${data_dir}/test.tok.bpe.en
# python ${BPEROOT}/apply_bpe.py -c ${model_dir}/ende30k.fastbpe.code  < ${data_dir}/train.tok.de > ${data_dir}/train.tok.bpe.de
# python ${BPEROOT}/apply_bpe.py -c ${model_dir}/ende30k.fastbpe.code  < ${data_dir}/valid.tok.de > ${data_dir}/valid.tok.bpe.de
# python ${BPEROOT}/apply_bpe.py -c ${model_dir}/ende30k.fastbpe.code  < ${data_dir}/test.tok.de > ${data_dir}/test.tok.bpe.de
