#!/usr/bin/env bash

model=sa_elmo
lang=en
save_dir=PATH_TO/data/models/${lang}/${model}
data_dir=PATH_TO/data/1b/training-monolingual.tokenized.shuffled/

mkdir -p $save_dir
echo "files saving to $save_dir"
export PYTHONPATH="$PYTHONPATH:."

#conda activate tf-gpu
CUDA_VISIBLE_DEVICES=0,1 python bin/train_elmo.py --save_dir $save_dir \
    --vocab_file ${data_dir}/vocab.txt  \
    --train_prefix ${data_dir}/'wiki_*' \
    > $save_dir/train.log 2>&1
