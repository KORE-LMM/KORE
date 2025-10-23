#!/bin/bash

# 默认参数
MODEL_PATH=""
SPLIT=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 --model-path <模型路径> --split <分割名称>"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [[ -z "$MODEL_PATH" ]]; then
    echo "错误: 必须提供 --model-path 参数"
    echo "用法: $0 --model-path <模型路径> --split <分割名称>"
    exit 1
fi

if [[ -z "$SPLIT" ]]; then
    echo "错误: 必须提供 --split 参数"
    echo "用法: $0 --model-path <模型路径> --split <分割名称>"
    exit 1
fi

# 从模型路径中提取CKPT（最后一个/后面的内容）
CKPT=$(basename "$MODEL_PATH")

echo "使用参数:"
echo "  MODEL_PATH: $MODEL_PATH"
echo "  SPLIT: $SPLIT"
echo "  CKPT: $CKPT"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader_infoseek \
        --model-path "$MODEL_PATH" \
        --question-file /home/bingxing2/ailab/scx6mh7/jkl/LLaVA_8_8_null_space/infoseek_sample_data/final_eval_infoseek_filtered_and_updated.jsonl \
        --image-folder /home/bingxing2/ailab/group/ai4bio/renyuchen/jkl/infoseek_data/downloaded_infoseek_images \
        --answers-file /home/bingxing2/ailab/scx6mh7/jkl/LLaVA_8_8_null_space/tool/infoseek_eavl/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --max_new_tokens 4 \
        --conv-mode vicuna_v1 &
done

wait

output_file=/home/bingxing2/ailab/scx6mh7/jkl/LLaVA_8_8_null_space/tool/infoseek_eavl/answers/$SPLIT/$CKPT/merge.jsonl

# 创建输出目录（如果不存在）
mkdir -p "$(dirname "$output_file")"

# 清空输出文件（如果存在）
> "$output_file"

# 循环遍历索引并连接每个文件
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /home/bingxing2/ailab/scx6mh7/jkl/LLaVA_8_8_null_space/tool/infoseek_eavl/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

echo "处理完成！结果保存在: $output_file"