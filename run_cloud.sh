#!/usr/bin/sh
MODEL=neulab/gpt2-finetuned-wikitext103
DSTORE_DIR=input/checkpoints/${MODEL}
OUTPUT_DIR=output/${MODEL}

python3 -u run_clm.py \
  --suffix_dfa \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --output_dir ${OUTPUT_DIR} \
  --do_eval --eval_subset validation \
  --dstore_dir=${DSTORE_DIR} \
  --min_factor_length=2 \
  --trunacte_dstore=1000

# For debugging, can add the following:
# --truncate_dstore=1000