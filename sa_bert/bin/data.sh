#!/usr/bin/env bash

for NB in {01..99}; do
FN="wiki_news.en-000${NB}-of-00100"
python create_pretraining_data.py \
  --input_file=PATH_TO/data/1b/training-monolingual.tokenized.shuffled/$FN \
  --output_file=PATH_TO/data/bert/1b/$FN \
  --vocab_file=PATH_TO/data/1b/training-monolingual.tokenized.shuffled/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
done;
