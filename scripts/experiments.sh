#!/usr/bin/sh

# === Suffix DFA experiments ===

OUTPUT_DIR=out/suffix_dfa-1
mkdir ${OUTPUT_DIR}
BUILD_METHOD=suffix_dfa MAX_POINTERS=1 OUTPUT_DIR=${OUTPUT_DIR} source scripts/run.sh

OUTPUT_DIR=out/suffix_dfa-1-f5
mkdir ${OUTPUT_DIR}
BUILD_METHOD=suffix_dfa MAX_POINTERS=1 MIN_LENGTH=5 OUTPUT_DIR=${OUTPUT_DIR} source scripts/run.sh

OUTPUT_DIR=out/suffix_dfa-1-t3
mkdir ${OUTPUT_DIR}
BUILD_METHOD=suffix_dfa MAX_POINTERS=1 TEMP=0.33 OUTPUT_DIR=${OUTPUT_DIR} source scripts/run.sh

# === Oracle experiments ===

OUTPUT_DIR=out/oracle-1-f5
mkdir ${OUTPUT_DIR}
BUILD_METHOD=oracle MAX_POINTERS=1 MIN_LENGTH=5 OUTPUT_DIR=${OUTPUT_DIR} source scripts/run.sh

OUTPUT_DIR=out/oracle-1-t5
mkdir ${OUTPUT_DIR}
BUILD_METHOD=oracle MAX_POINTERS=1 TEMP=0.2 OUTPUT_DIR=${OUTPUT_DIR} source scripts/run.sh

# OUTPUT_DIR=out/linear
# mkdir ${OUTPUT_DIR}
# BUILD_METHOD=linear MAX_POINTERS=-1 NO_FAIL=False OUTPUT_DIR=${OUTPUT_DIR} source scripts/run.sh

# OUTPUT_DIR=out/oracle-1-no_fail
# mkdir ${OUTPUT_DIR}
# BUILD_METHOD=oracle MAX_POINTERS=1 NO_FAIL=True OUTPUT_DIR=${OUTPUT_DIR} source scripts/run.sh

# OUTPUT_DIR=out/oracle-1
# mkdir ${OUTPUT_DIR}
# BUILD_METHOD=oracle MAX_POINTERS=1 NO_FAIL=False OUTPUT_DIR=${OUTPUT_DIR} source scripts/run.sh
