#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b"
SPLIT="plain"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader_infoseek \
        --model-path /home/bingxing2/ailab/scx6mh7/jkl/ckpt_sum/vlm_ckpt/llava-v1.5-7b \
        --question-file /home/bingxing2/ailab/scx6mh7/jkl/LLaVA_8_8_null_space/infoseek_sample_data/final_eval_infoseek_filtered_and_updated.jsonl \
        --image-folder /home/bingxing2/ailab/group/ai4bio/renyuchen/jkl/infoseek_data/downloaded_infoseek_images \
        --answers-file /home/bingxing2/ailab/scx6mh7/jkl/LLaVA_8_8_null_space/tool/infoseek_eavl/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait






output_file=/home/bingxing2/ailab/scx6mh7/jkl/LLaVA_8_8_null_space/tool/infoseek_eavl/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /home/bingxing2/ailab/scx6mh7/jkl/LLaVA_8_8_null_space/tool/infoseek_eavl/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

        # --question-file /home/bingxing2/ailab/group/ai4bio/renyuchen/jkl/MME/llava_mme.jsonl \
        # --image-folder /home/bingxing2/ailab/group/ai4bio/renyuchen/jkl/MME/MME_Benchmark_release_version/MME_Benchmark \

        # --question-file /home/bingxing2/ailab/scx6mh7/jkl/LLaVA_8_8_null_space/infoseek_sample_data/final_eval_infoseek_filtered_and_updated.jsonl \
        # --image-folder /home/bingxing2/ailab/group/ai4bio/renyuchen/jkl/infoseek_data/downloaded_infoseek_images \