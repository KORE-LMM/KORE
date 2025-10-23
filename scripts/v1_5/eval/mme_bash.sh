#!/bin/bash

# 检查是否提供了model-path参数
if [ $# -eq 0 ]; then
    echo "使用方法: $0 <model-path>"
    echo "示例: $0 /media/raid/workspace/jiangkailin/LLaVA/merge_llava_null_space"
    exit 1
fi

MODEL_PATH="$1"

# 从model-path中提取最后一个路径组件作为experiment名称
# 提取两个名称
EXPERIMENT_NAME=$(basename "$MODEL_PATH")
EXPERIMENT_NAME2=$(basename "$(dirname "$MODEL_PATH")")

# 组合成最终名称：EXPERIMENT_NAME2_EXPERIMENT_NAME
FINAL_EXPERIMENT_NAME="${EXPERIMENT_NAME2}_${EXPERIMENT_NAME}"

echo "使用模型路径: $MODEL_PATH"
echo "提取的实验名称: $EXPERIMENT_NAME"

# 运行模型评估
python -m llava.eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
    --question-file /home/bingxing2/ailab/group/ai4bio/renyuchen/jkl/MME/llava_mme.jsonl \
    --image-folder /home/bingxing2/ailab/group/ai4bio/renyuchen/jkl/MME/MME_Benchmark_release_version/MME_Benchmark \
    --answers-file "/home/bingxing2/ailab/group/ai4bio/renyuchen/jkl/MME/answers/$FINAL_EXPERIMENT_NAME.jsonl" \
    --temperature 0 \
    --conv-mode vicuna_v1

cd /home/bingxing2/ailab/group/ai4bio/renyuchen/jkl/MME

# 使用提取的实验名称
python convert_answer_to_mme.py --experiment "$FINAL_EXPERIMENT_NAME"

cd /home/bingxing2/ailab/group/ai4bio/renyuchen/jkl/MME/eval_tool

# 使用提取的实验名称
python calculation.py --results_dir "answers/$FINAL_EXPERIMENT_NAME"