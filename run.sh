#!/usr/bin/sh
MODEL=neulab/gpt2-finetuned-wikitext103

python3 -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --output_dir checkpoints/${MODEL} \
  --do_eval --eval_subset validation \
  --dstore_dir checkpoints/${MODEL} --dfa_retomaton
