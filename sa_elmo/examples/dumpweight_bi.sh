#!/usr/bin/env bash

model=sa_elmo_bi
lang=en_de
save_dir=PATH_TO/data/models/${lang}/${model}
data_dir=PATH_TO/data/processed/${lang}/

mkdir -p $save_dir
echo "files saving to $save_dir"
export PYTHONPATH="$PYTHONPATH:."

CUDA_VISIBLE_DEVICES=7 python bin/dump_weights.py --save_dir $save_dir --outfile $save_dir/weights.hdf5
