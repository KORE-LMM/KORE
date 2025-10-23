
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


CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_null/build_adapter_mllm.py \
    --model_id "liuhaotian/llava-v1.5-7b" \
    --seed $SEED \
    --singular_aware \
    --r $RANK \
    --save_model \
    --n_samples_per_dataset $N_SAMPLES \
    --save_sampled_data True \
    --dataset_names $DATASET_NAMES \
    --onevision_image_dir  ov_data \
    --onevision_local_dir LLaVA-OneVision-Data \
    --onevision_sources Geometry3K\(MathV360K\),iiit5k,VizWiz\(MathV360K\),FigureQA\(MathV360K\) \
    --output_file ov_data/benchmark_seed_data/onevision/benchmark_original_data_rank${RANK}_${DATASET_NAMES_UNDERSCORE}_pre_${N_SAMPLES}/${RANK}_${DATASET_NAMES_UNDERSCORE}_pre_${N_SAMPLES}.jsonl \
    --save_path ov_data/lora_null_sample_ckpt/save_LoRA_Null_adapter_llava_7b_PT_${RANK}_ov_pre_${N_SAMPLES} \
    --temperature 0 \
    --conv-mode llava_v1