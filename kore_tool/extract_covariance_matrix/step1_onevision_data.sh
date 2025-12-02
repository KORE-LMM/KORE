
#!/usr/bin/env bash

# Default arguments
RANK=256         # -r: rank
N_SAMPLES=64     # -n: number of samples per dataset
SEED=233         # -s: random seed
DATASET_NAMES="onevision"

# Parse command line arguments: -r -n -s
while getopts "r:n:s:" opt; do
  case "$opt" in
    r) RANK="$OPTARG" ;;
    n) N_SAMPLES="$OPTARG" ;;
    s) SEED="$OPTARG" ;;
    *)
      echo "Usage: $0 [-r RANK] [-n N_SAMPLES] [-s SEED]"
      exit 1
      ;;
  esac
done

DATASET_NAMES_UNDERSCORE=$(echo $DATASET_NAMES | tr ' ' '_')

CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_null/build_adapter_mllm.py \
    --model_id "vlm_ckpt/Qwen2.5-VL-7B-Instruct" \
    --seed $SEED \
    --singular_aware \
    --r $RANK \
    --save_model \
    --n_samples_per_dataset $N_SAMPLES \
    --save_sampled_data True \
    --dataset_names $DATASET_NAMES \
    --onevision_image_dir  vlm_ckpt/ov_data \
    --onevision_local_dir vlm_ckpt/LLaVA-OneVision-Data \
    --onevision_sources Geometry3K\(MathV360K\),iiit5k,VizWiz\(MathV360K\),FigureQA\(MathV360K\) \
    --output_file vlm_ckpt/ov_data/qwen2_5_vl/benchmark_original_data_rank${RANK}_${DATASET_NAMES_UNDERSCORE}_pre_${N_SAMPLES}/${RANK}_${DATASET_NAMES_UNDERSCORE}_pre_${N_SAMPLES}.jsonl \
    --save_path sample_ckpt/qwen2_5_vl/save_LoRA_Null_adapter_qwen2_5_vl_PT_${RANK}_ov_pre_${N_SAMPLES} \
    --temperature 0 \
    --conv-mode llava_v1