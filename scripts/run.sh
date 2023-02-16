#!/usr/bin/sh
MODEL=neulab/gpt2-finetuned-wikitext103
DATASET=wikitext-103-raw-v1
# MODEL=neulab/distilgpt2

# Important settings.
# CHAIN=False
# MIN_LENGTH=2
# OUTPUT_DIR=out/suffix-1
# MAX_POINTERS=1

# Other settings.
METHOD=suffix_dfa
MIN_KNNS=10000
INITIAL=True
NO_LOAD=True

if [ -z "$NO_FAIL" ]
then
  echo "Empty \$NO_FAIL, set to False"
  NO_FAIL=False
fi

if [ -z "$MIN_LENGTH" ]
then
  echo "Empty \$MIN_LENGTH, set to 2"
  MIN_LENGTH=2
fi

if [ -z "$TEMP" ]
then
  echo "Empty \$TEMP, set to 1.0"
  TEMP=1.0
fi

python3 -u run_clm.py \
  --${METHOD} \
  --model_name_or_path ${MODEL} \
  --no_load_keys=${NO_LOAD} \
  --dataset_name wikitext --dataset_config_name ${DATASET} \
  --output_dir=${OUTPUT_DIR} \
  --do_eval --eval_subset validation \
  --dstore_dir checkpoints/${MODEL} \
  --min_factor_length=${MIN_LENGTH} \
  --add_initial=${INITIAL} \
  --min_knns=${MIN_KNNS} \
  --max_knns=1024 \
  --max_states=1024 \
  --max_pointers=${MAX_POINTERS} \
  --build_method=${BUILD_METHOD} \
  --no_failures=${NO_FAIL} \
  --knn_temp=${TEMP}
  # --output_dir checkpoints/${MODEL} \
  # --truncate_dstore=1000
  # --eval_limit=-1 \
  # --pointer_log_path=trace/${METHOD}.txt \
  # --count_plot_path=trace/${METHOD}.png \
  # --max_eval_samples={$MAX_EVAL}
