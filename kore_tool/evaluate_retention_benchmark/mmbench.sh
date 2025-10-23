#!/bin/bash

# Initialize variables
MODEL_PATH=""
SPLIT="mmbench_dev_20230712"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -s|--split)
            SPLIT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 -m MODEL_PATH [-s SPLIT]"
            echo "  -m, --model-path    Path to the model checkpoint"
            echo "  -s, --split        Dataset split (default: mmbench_dev_20230712)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Example: $0 -m /media/raid/workspace/jiangkailin/LLaVA/merge_llava_null_space"
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 -m MODEL_PATH [-s SPLIT]"
            echo "Use -h or --help for more information"
            exit 1
            ;;
    esac
done

# Check if model-path is provided
if [ -z "$MODEL_PATH" ]; then
    echo "Error: Model path is required"
    echo "Usage: $0 -m MODEL_PATH [-s SPLIT]"
    echo "Use -h or --help for more information"
    exit 1
fi

# Validate model path
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory does not exist: $MODEL_PATH"
    exit 1
fi

# Extract experiment name from model-path
# Extract three names from the path hierarchy
FINAL_EXPERIMENT_NAME="$(basename "$(dirname "$(dirname "$MODEL_PATH")")")_$(basename "$(dirname "$MODEL_PATH")")_$(basename "$MODEL_PATH")"
echo "$FINAL_EXPERIMENT_NAME"

echo "Using model path: $MODEL_PATH"
echo "Extracted experiment name: $FINAL_EXPERIMENT_NAME"

python -m llava.eval.model_vqa_mmbench \
    --model-path "$MODEL_PATH" \
    --question-file /hkfs/work/workspace/scratch/lmu_chd4938-MINED_26/MINED_26/data_ckpt/dataset/mmbench/$SPLIT.tsv \
    --answers-file "/hkfs/work/workspace/scratch/lmu_chd4938-MINED_26/MINED_26/data_ckpt/dataset/mmbench/answers/$SPLIT/$FINAL_EXPERIMENT_NAME.jsonl" \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /hkfs/work/workspace/scratch/lmu_chd4938-MINED_26/MINED_26/data_ckpt/dataset/mmbench/answers_upload/$SPLIT

python /hkfs/work/workspace/scratch/lmu_chd4938-MINED_26/LLaVA_8_8_null_space/scripts/convert_mmbench_for_submission.py \
    --annotation-file /hkfs/work/workspace/scratch/lmu_chd4938-MINED_26/MINED_26/data_ckpt/dataset/mmbench/$SPLIT.tsv \
    --result-dir /hkfs/work/workspace/scratch/lmu_chd4938-MINED_26/MINED_26/data_ckpt/dataset/mmbench/answers/$SPLIT \
    --upload-dir /hkfs/work/workspace/scratch/lmu_chd4938-MINED_26/MINED_26/data_ckpt/dataset/mmbench/answers_upload/$SPLIT \
    --experiment "$FINAL_EXPERIMENT_NAME"