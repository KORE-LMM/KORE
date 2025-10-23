#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path /media/raid/workspace/jiangkailin/LLaVA/merge_llava_null_space \
    --question-file /media/raid/workspace/jiangkailin/LLaVA/playground/data/eval/MME/llava_mme.jsonl \
    --image-folder /media/raid/workspace/jiangkailin/LLaVA/playground/data/eval/MME/MME_Benchmark_release_version/MME_Benchmark \
    --answers-file ./playground/data/eval/MME/answers/mllm_save_LoRA_Null_adapter_llava_PT_128_math_Null_v1_merged.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment llava-v1.5-13b

cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-13b


# python -m llava.eval.model_vqa_loader \
#     --model-path liuhaotian/llava-v1.5-7b \
#     --question-file /data/jhb_data/codes/LLaVA/playground/data/eval/MME/llava_mme.jsonl \
#     --image-folder /data/jhb_data/codes/LLaVA/playground/data/eval/MME \
#     --answers-file /data/jhb_data/codes/LLaVA/playground/data/eval/MME/answers/answers.jsonl \
#     --temperature 0 \
#     --conv-mode llava_v1