#!/bin/bash

# 检查 CUDA_VISIBLE_DEVICES 环境变量是否设置
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES is not set. Please set it before running the script."
    exit 1
fi

# 检查是否通过命令行传入 CKPT_PATH、OUTPUT_DIR_PREFIX 和 QUESTION_FILE
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 CKPT_PATH OUTPUT_DIR_PREFIX QUESTION_FILE"
    exit 1
fi

# 获取检查点路径
CKPT_PATH="$1"

# 获取输出文件路径前缀
OUTPUT_DIR_PREFIX="$2"

# 获取问题文件路径
QUESTION_FILE="$3"

# 确认 CUDA_VISIBLE_DEVICES 环境变量并解析 GPU 列表
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# 计算 GPU 块数
CHUNKS=${#GPULIST[@]}

# 从路径中提取 CKPT 名称
CKPT=$(basename "$CKPT_PATH")

# 根据 CKPT 值设置 SPLIT
if [ "$CKPT" == "merge_PR_4_epoch_7_llava_7b_lora" ]; then
    SPLIT="all_Phase4_eval_vqa"
else
    SPLIT="eval_vqa"
fi

# 配置 output_dir 和 output_file
output_dir="$OUTPUT_DIR_PREFIX/$SPLIT/$CKPT"
output_file="$output_dir/merge.jsonl"

# 创建输出目录（如果不存在）
mkdir -p "$output_dir"

# 为每个分块任务启动进程
for ((IDX=0; IDX<CHUNKS; IDX++)); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path "$CKPT_PATH" \
        --question-file "$QUESTION_FILE" \
        --image-folder /hkfs/work/workspace/scratch/lmu_chd4938-MINED_26/MINED_26/data_ckpt/dataset/EVOKE \
        --answers-file "$output_dir/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

# 等待所有分块任务完成
wait

# 合并输出文件
> "$output_file"
for ((IDX=0; IDX<CHUNKS; IDX++)); do
    cat "$output_dir/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
done

# # 执行转换脚本
# python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT