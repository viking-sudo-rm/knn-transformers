#!/usr/bin/sh
MODEL=neulab/gpt2-finetuned-wikitext103
# TRUNCATE=1000
TRUNCATE=-1
RETO=False

python3 -u run_clm.py \
  --suffix_dfa \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --output_dir checkpoints/${MODEL} \
  --do_eval --eval_subset validation \
  --dstore_dir checkpoints/${MODEL} \
  --min_factor_length=2 \
  --truncate_dstore=${TRUNCATE} \
  --retomaton=${RETO} \
  --min_knns=10000 \
  --solid_only
