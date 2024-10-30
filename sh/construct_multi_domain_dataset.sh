src_data_dir=./nmt/data/multi-domain
tgt_data_dir=./nmt/data/multi-domain-mix

awk '{print "" $0}' ${src_data_dir}/it/test_copy.de ${src_data_dir}/koran/test_copy.de \
${src_data_dir}/medical/test_copy.de ${src_data_dir}/law/test_copy.de ${src_data_dir}/subtitles/test_copy.de \
> ${tgt_data_dir}/test_copy.de

awk '{print "" $0}' ${src_data_dir}/it/test_copy.en ${src_data_dir}/koran/test_copy.en \
${src_data_dir}/medical/test_copy.en ${src_data_dir}/law/test_copy.en ${src_data_dir}/subtitles/test_copy.en \
> ${tgt_data_dir}/test_copy.en

awk '{print "" $0}' ${src_data_dir}/it/valid_copy.de ${src_data_dir}/koran/valid_copy.de \
${src_data_dir}/medical/valid_copy.de ${src_data_dir}/law/valid_copy.de ${src_data_dir}/subtitles/valid_copy.de \
> ${tgt_data_dir}/valid_copy.de

awk '{print "" $0}' ${src_data_dir}/it/valid_copy.en ${src_data_dir}/koran/valid_copy.en \
${src_data_dir}/medical/valid_copy.en ${src_data_dir}/law/valid_copy.en ${src_data_dir}/subtitles/valid_copy.en \
> ${tgt_data_dir}/valid_copy.en

awk '{print "" $0}' ${src_data_dir}/it/train_copy.de ${src_data_dir}/koran/train_copy.de \
${src_data_dir}/medical/train_copy.de ${src_data_dir}/law/train_copy.de ${src_data_dir}/subtitles/train_copy.de \
> ${tgt_data_dir}/train_copy.de

awk '{print "" $0}' ${src_data_dir}/it/train_copy.en ${src_data_dir}/koran/train_copy.en \
${src_data_dir}/medical/train_copy.en ${src_data_dir}/law/train_copy.en ${src_data_dir}/subtitles/train_copy.en \
> ${tgt_data_dir}/train_copy.en