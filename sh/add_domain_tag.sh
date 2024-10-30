data_dir=./nmt/data/general-tag

awk '{print "<general> " $0}' ${data_dir}/test.de > ${data_dir}/test_tag.de

awk '{print "<general> " $0}' ${data_dir}/valid.de > ${data_dir}/valid_tag.de

awk '{print "<general> " $0}' ${data_dir}/train.de > ${data_dir}/train_tag.de

awk '{print "" $0}' ${data_dir}/test.en > ${data_dir}/test_tag.en

awk '{print "" $0}' ${data_dir}/valid.en > ${data_dir}/valid_tag.en

awk '{print "" $0}' ${data_dir}/train.en > ${data_dir}/train_tag.en

# src_data_dir=./nmt/data/multi-domain-tag-gather
# tgt_data_dir=./nmt/data/multi-domain-tag-gather/gather

# awk '{print "" $0}' ${src_data_dir}/it/test_tag.de ${src_data_dir}/koran/test_tag.de \
# ${src_data_dir}/medical/test_tag.de ${src_data_dir}/law/test_tag.de ${src_data_dir}/subtitles/test_tag.de \
# > ${tgt_data_dir}/test_tag.de

