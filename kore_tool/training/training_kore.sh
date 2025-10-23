#!/bin/bash

# Parse command line arguments
DATA_PATH=""
OUTPUT_DIR=""
NUM_TRAIN_EPOCHS=""
PER_DEVICE_TRAIN_BATCH_SIZE=""
SWANLAB_PROJECT=""
SWANLAB_EXPERIMENT_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_path|--data)
            DATA_PATH="$2"
            shift 2
            ;;
        --output_dir|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_train_epochs|--epochs)
            NUM_TRAIN_EPOCHS="$2"
            shift 2
            ;;
        --per_device_train_batch_size|--batch_size)
            PER_DEVICE_TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --swanlab_project|--project)
            SWANLAB_PROJECT="$2"
            shift 2
            ;;
        --swanlab_experiment_name|--exp_name)
            SWANLAB_EXPERIMENT_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 --data_path <data_path> --output_dir <output_dir> --epochs <num_train_epochs> --batch_size <per_device_train_batch_size> [--swanlab_project <project_name>] [--swanlab_experiment_name <experiment_name>]"
            exit 1
            ;;
    esac
done

# Check required parameters
if [ -z "$DATA_PATH" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$NUM_TRAIN_EPOCHS" ] || [ -z "$PER_DEVICE_TRAIN_BATCH_SIZE" ]; then
    echo "Error: Missing required parameters"
    echo "Usage: $0 --data_path <data_path> --output_dir <output_dir> --epochs <num_train_epochs> --batch_size <per_device_train_batch_size> [--swanlab_project <project_name>] [--swanlab_experiment_name <experiment_name>]"
    echo "Example: $0 --data /path/to/data.json --output /path/to/output --epochs 1 --batch_size 6 --project my-project --exp_name experiment-v1"
    exit 1
fi

# Validate parameters
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: data_path file does not exist: $DATA_PATH"
    exit 1
fi

if [ ! -d "$(dirname "$OUTPUT_DIR")" ]; then
    echo "Error: Parent directory of output_dir does not exist: $(dirname "$OUTPUT_DIR")"
    exit 1
fi

echo "Using parameters:"
echo "  data_path: $DATA_PATH"
echo "  output_dir: $OUTPUT_DIR"
echo "  num_train_epochs: $NUM_TRAIN_EPOCHS"
echo "  per_device_train_batch_size: $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "  swanlab_project: ${SWANLAB_PROJECT:-'llava-plain-lora (default)'}"
echo "  swanlab_experiment_name: ${SWANLAB_EXPERIMENT_NAME:-'llava-plain-lora-epoch3 (default)'}"
echo ""


SWANLAB_ARGS=""
if [ -n "$SWANLAB_PROJECT" ]; then
    SWANLAB_ARGS="$SWANLAB_ARGS --swanlab_project $SWANLAB_PROJECT"
fi
if [ -n "$SWANLAB_EXPERIMENT_NAME" ]; then
    SWANLAB_ARGS="$SWANLAB_ARGS --swanlab_experiment_name $SWANLAB_EXPERIMENT_NAME"
fi


echo "Starting DeepSpeed training..."
deepspeed llava/train/train_mem_v2.py \
    --lora_null_enable True --lora_null_v2 True --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path save_LoRA_Null_adapter_llava_PT_235_ov_pre_64 \
    --version v1 \
    --data_path "$DATA_PATH" \
    --image_folder KORE-74K \
    --vision_tower clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    $SWANLAB_ARGS