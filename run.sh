#!/usr/bin/sh
MODEL=neulab/gpt2-finetuned-wikitext103

python3 -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --output_dir checkpoints/${MODEL} \
  --do_eval --eval_subset validation \
  --dstore_dir checkpoints/${MODEL} \
  --dfa_retomaton \
  --min_factor_length=2 \
  # The following should be enabled for debugging.
  # --truncate_dstore=1000 \
  --cache_path=cached
