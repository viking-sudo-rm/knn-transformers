#!/usr/bin/sh
MODEL=neulab/gpt2-finetuned-wikitext103
DSTORE_DIR=cloud/input/checkpoints/${MODEL}
OUTPUT_DIR=cloud/output/${MODEL}

python3 -u run_clm.py \
  --suffix_dfa \
  --do_eval --eval_subset validation \
  --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --model_name_or_path ${MODEL} \
  --output_dir ${OUTPUT_DIR} \
  --dstore_dir=${DSTORE_DIR} \
  --min_factor_length=2 \
  --truncate_dstore=1000

# For debugging, can add the following:
# --truncate_dstore=1000