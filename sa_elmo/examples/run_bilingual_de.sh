#!/usr/bin/env bash

model=sa_elmo_bi
lang=en_de
save_dir=PATH_TO/data/models/${lang}/${model}
data_dir=PATH_TO/data/processed

mkdir -p $save_dir
echo "files saving to $save_dir"
export PYTHONPATH="$PYTHONPATH:."

#conda activate tf-gpu
CUDA_VISIBLE_DEVICES=6,7 python bin/train_bilingual.py --save_dir $save_dir \
    --vocab_file ${data_dir}/dict/de_en/vocab.txt  \
    --dictionary ${data_dir}/dict/de_en/de_en.txt  \
    --train_prefix1 ${data_dir}/en_sample/'wiki_*' \
    --train_prefix2 ${data_dir}/de_sample/'wiki_*' \
    > $save_dir/train.log 2>&1
