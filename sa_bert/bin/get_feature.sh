#!/usr/bin/env bash

DATA_DIR=PATH_TO/tmp/pca
INPUT_PATH=$DATA_DIR/tmp.txt
OUTPUT_PATH=$DATA_DIR/output.json
#BERT_BASE_DIR='PATH_TO/data/models_bert/original/uncased_L-12_H-768_A-12'
BERT_BASE_DIR='PATH_TO/data/models_bert/1b/hist'
VOCAB=$BERT_BASE_DIR/vocab.txt
CONFIG_PATH=$BERT_BASE_DIR/bert_config.json
CKPT_PATH=$BERT_BASE_DIR/'model.ckpt-375000'


CUDA_VISIBLE_DEVICES=2 python extract_features.py \
  --input_file=$INPUT_PATH \
  --output_file=$OUTPUT_PATH \
  --vocab_file=$VOCAB \
  --bert_config_file=$CONFIG_PATH \
  --init_checkpoint=$CKPT_PATH \
  --layers=-1 \
  --max_seq_length=128 \
  --batch_size=8
