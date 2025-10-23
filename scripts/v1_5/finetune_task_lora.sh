#!/bin/bash

# 解析命令行参数
DATA_PATH=""
OUTPUT_DIR=""
NUM_TRAIN_EPOCHS=""
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
        --swanlab_project|--project)
            SWANLAB_PROJECT="$2"
            shift 2
            ;;
        --swanlab_experiment_name|--exp_name)
            SWANLAB_EXPERIMENT_NAME="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 --data_path <data_path> --output_dir <output_dir> --epochs <num_train_epochs> [--swanlab_project <project_name>] [--swanlab_experiment_name <experiment_name>]"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$DATA_PATH" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$NUM_TRAIN_EPOCHS" ]; then
    echo "错误: 缺少必需参数"
    echo "用法: $0 --data_path <data_path> --output_dir <output_dir> --epochs <num_train_epochs> [--swanlab_project <project_name>] [--swanlab_experiment_name <experiment_name>]"
    echo "示例: $0 --data /path/to/data.json --output /path/to/output --epochs 1 --project my-project --exp_name experiment-v1"
    exit 1
fi

# 验证参数
if [ ! -f "$DATA_PATH" ]; then
    echo "错误: data_path 文件不存在: $DATA_PATH"
    exit 1
fi

if [ ! -d "$(dirname "$OUTPUT_DIR")" ]; then
    echo "错误: output_dir 的父目录不存在: $(dirname "$OUTPUT_DIR")"
    exit 1
fi

echo "使用参数:"
echo "  data_path: $DATA_PATH"
echo "  output_dir: $OUTPUT_DIR"
echo "  num_train_epochs: $NUM_TRAIN_EPOCHS"
echo "  swanlab_project: ${SWANLAB_PROJECT:-'llava-plain-lora (默认)'}"
echo "  swanlab_experiment_name: ${SWANLAB_EXPERIMENT_NAME:-'llava-plain-lora-epoch3 (默认)'}"
echo ""

# 构建 SwanLab 参数
SWANLAB_ARGS=""
if [ -n "$SWANLAB_PROJECT" ]; then
    SWANLAB_ARGS="$SWANLAB_ARGS --swanlab_project $SWANLAB_PROJECT"
fi
if [ -n "$SWANLAB_EXPERIMENT_NAME" ]; then
    SWANLAB_ARGS="$SWANLAB_ARGS --swanlab_experiment_name $SWANLAB_EXPERIMENT_NAME"
fi

# 启动 DeepSpeed 训练
echo "开始 DeepSpeed 训练..."
deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /hkfs/work/workspace/scratch/lmu_chd4938-MINED_26/MINED_26/data_ckpt/vlm_ckpt/llava-v1.5-7b \
    --version v1 \
    --data_path "$DATA_PATH" \
    --image_folder /hkfs/work/workspace/scratch/lmu_chd4938-MINED_26/MINED_26/data_ckpt/dataset/lora_null \
    --vision_tower /hkfs/work/workspace/scratch/lmu_chd4938-MINED_26/MINED_26/data_ckpt/vlm_ckpt/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size 50 \
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
    --report_to wandb \
    $SWANLAB_ARGS