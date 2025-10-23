#!/bin/bash

# Check if CUDA_VISIBLE_DEVICES environment variable is set
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES is not set. Please set it before running the script."
    exit 1
fi

# Initialize variables
CKPT_PATH=""
OUTPUT_DIR_PREFIX=""
QUESTION_FILE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--checkpoint)
            CKPT_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR_PREFIX="$2"
            shift 2
            ;;
        -q|--question)
            QUESTION_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 -c CKPT_PATH -o OUTPUT_DIR_PREFIX -q QUESTION_FILE"
            echo "  -c, --checkpoint    Path to the model checkpoint"
            echo "  -o, --output        Output directory prefix"
            echo "  -q, --question      Path to the question file"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 -c CKPT_PATH -o OUTPUT_DIR_PREFIX -q QUESTION_FILE"
            exit 1
            ;;
    esac
done

# Check if all required parameters are provided
if [ -z "$CKPT_PATH" ] || [ -z "$OUTPUT_DIR_PREFIX" ] || [ -z "$QUESTION_FILE" ]; then
    echo "Error: Missing required parameters"
    echo "Usage: $0 -c CKPT_PATH -o OUTPUT_DIR_PREFIX -q QUESTION_FILE"
    echo "Use -h or --help for more information"
    exit 1
fi

# Validate parameters
if [ ! -d "$CKPT_PATH" ]; then
    echo "Error: Checkpoint directory does not exist: $CKPT_PATH"
    exit 1
fi

if [ ! -f "$QUESTION_FILE" ]; then
    echo "Error: Question file does not exist: $QUESTION_FILE"
    exit 1
fi

# Confirm CUDA_VISIBLE_DEVICES environment variable and parse GPU list
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# Calculate number of GPU chunks
CHUNKS=${#GPULIST[@]}

# Extract CKPT name from path
CKPT=$(basename "$CKPT_PATH")

# Set SPLIT based on CKPT value
if [ "$CKPT" == "merge_PR_4_epoch_7_llava_7b_lora" ]; then
    SPLIT="all_Phase4_eval_vqa"
else
    SPLIT="eval_vqa"
fi

# Configure output_dir and output_file
output_dir="$OUTPUT_DIR_PREFIX/$SPLIT/$CKPT"
output_file="$output_dir/merge.jsonl"

# Create output directory (if it doesn't exist)
mkdir -p "$output_dir"

# Start processes for each chunk task
for ((IDX=0; IDX<CHUNKS; IDX++)); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path "$CKPT_PATH" \
        --question-file "$QUESTION_FILE" \
        --image-folder dataset/EVOKE \
        --answers-file "$output_dir/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

# Wait for all chunk tasks to complete
wait

# Merge output files
> "$output_file"
for ((IDX=0; IDX<CHUNKS; IDX++)); do
    cat "$output_dir/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
done