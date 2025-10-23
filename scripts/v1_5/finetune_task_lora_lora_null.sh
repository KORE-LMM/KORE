

#!/bin/bash

deepspeed llava/train/train_mem_v2.py \
    --lora_null_enable True --lora_null_v1 True --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /home/bingxing2/ailab/scx6mh7/jkl/ckpt_sum/vlm_ckpt/llava-v1.5-7b \
    --version v1 \
    --data_path /home/bingxing2/ailab/scx6mh7/jkl/LLaVA_8_8_null_space/train_data/random_train_data_seed42.json \
    --image_folder /home/bingxing2/ailab/group/ai4bio/renyuchen/jkl/lora_null \
    --vision_tower /home/bingxing2/ailab/scx6mh7/jkl/ckpt_sum/vlm_ckpt/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /home/bingxing2/ailab/group/ai4bio/renyuchen/jkl/lora_null_train_ckpt/vanilla_lora/llava_lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
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
    --report_to none