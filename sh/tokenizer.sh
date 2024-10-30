data_dir=./store/data/multi-domain/law
SCRIPTS=./mosesdecoder/scripts
TOKENIZER=${SCRIPTS}/tokenizer/tokenizer.perl

${TOKENIZER} -l en < ${data_dir}/train_copy.en > ${data_dir}/train.tok.en
${TOKENIZER} -l en < ${data_dir}/valid_copy.en > ${data_dir}/valid.tok.en
${TOKENIZER} -l en < ${data_dir}/test_copy.en > ${data_dir}/test.tok.en
${TOKENIZER} -l de < ${data_dir}/train_copy.de > ${data_dir}/train.tok.de
${TOKENIZER} -l de < ${data_dir}/valid_copy.de > ${data_dir}/valid.tok.de
${TOKENIZER} -l de < ${data_dir}/test_copy.de > ${data_dir}/test.tok.de

