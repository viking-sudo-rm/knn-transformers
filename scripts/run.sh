#!/usr/bin/sh
MODEL=neulab/gpt2-finetuned-wikitext103
# MODEL=neulab/distilgpt2

CHAIN=True
MIN_LENGTH=0
INITIAL=False

python3 -u run_clm.py \
  --suffix_dfa \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --output_dir checkpoints/${MODEL} \
  --do_eval --eval_subset validation \
  --dstore_dir checkpoints/${MODEL} \
  --min_factor_length=${MIN_LENGTH} \
  --add_initial=${INITIAL} \
  --truncate_dstore=-1 \
  --linear_dfa=${CHAIN} \
  --min_knns=10000
