#!/usr/bin/env bash

OUTDIR=PATH_TO/data/models_bert/en_1b_1
echo "outdir $OUTDIR"
mkdir -p $OUTDIR
mkdir -p $OUTDIR/warmup

CUDA_VISIBLE_DEVICES=3 python run_pretraining.py \
  --input_file=PATH_TO/data/bert/1b/wiki_news* \
  --output_dir=$OUTDIR \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=./bin/bert_config.json \
  --train_batch_size=128 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=1000000 \
  --num_warmup_steps=20000 \
  --learning_rate=1e-4 \
  > $OUTDIR/train.log 2>&1

#--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
