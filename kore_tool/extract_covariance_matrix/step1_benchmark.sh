#!/bin/bash

# Accept parameters from the command line
while getopts "d:n:r:s:" opt; do
  case ${opt} in
    d) DATASET_NAMES=$OPTARG ;;
    n) N_SAMPLES=$OPTARG ;;
    r) RANK=$OPTARG ;;
    s) SEED=$OPTARG ;;
    \?) echo "Usage: cmd [-d dataset_names] [-n n_samples] [-r rank] [-s seed]"; exit 1 ;;
  esac
done

# Convert dataset names to underscore format
DATASET_NAMES_UNDERSCORE=$(echo $DATASET_NAMES | tr ' ' '_')

# Run the Python script with passed parameters
CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_null/build_adapter_mllm.py \
    --model_id "liuhaotian/llava-v1.5-7b" \
    --seed $SEED \
    --save_model \
    --singular_aware \
    --r $RANK \
    --dataset_names $DATASET_NAMES \
    --n_samples_per_dataset $N_SAMPLES \
    --data_root dataset/vlmeval \
    --save_sampled_data True \
    --output_file "null_space_sample_specific_know/benchmark_seed_data/benchmark_original_data_rank${RANK}_${DATASET_NAMES_UNDERSCORE}_pre_${N_SAMPLES}/${RANK}_${DATASET_NAMES_UNDERSCORE}_pre_${N_SAMPLES}.jsonl" \
    --save_path null_space_sample_specific_know/lora_null_sample_ckpt_v2/llava_rank${RANK}_${DATASET_NAMES_UNDERSCORE}_pre_${N_SAMPLES} \
    --temperature 0 \
    --conv-mode llava_v1





    