#!/usr/bin/sh
MODEL=neulab/gpt2-finetuned-wikitext103
# MODEL=neulab/distilgpt2

METHOD=suffix_dfa
CHAIN=False
MIN_LENGTH=2
MIN_KNNS=10000
INITIAL=False
NO_LOAD=True

python3 -u run_clm.py \
  --${METHOD} \
  --model_name_or_path ${MODEL} \
  --no_load_keys=${NO_LOAD} \
  --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --output_dir checkpoints/${MODEL} \
  --do_eval --eval_subset validation \
  --dstore_dir checkpoints/${MODEL} \
  --min_factor_length=${MIN_LENGTH} \
  --add_initial=${INITIAL} \
  --linear_dfa=${CHAIN} \
  --min_knns=${MIN_KNNS} \
  --max_knns=1024 \
  --max_states=1024
  # --truncate_dstore=1000
  # --eval_limit=-1 \
  # --pointer_log_path=trace/${METHOD}.txt \
  # --count_plot_path=trace/${METHOD}.png
