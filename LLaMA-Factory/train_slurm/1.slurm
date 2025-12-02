#!/bin/bash
#SBATCH --job-name qwen-replay                 
#SBATCH --gres gpu:4  
#SBATCH --partition accelerated                
#SBATCH --time 2:30:00                      
#SBATCH --output /hkfs/work/workspace/scratch/lmu_chd4938-MINED_26/log/qwen-vl/replay/replay.log





conda init
source activate qwen_2_5_train

which python


echo "Allocated nodes: $SLURM_JOB_NODELIST"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

cd /hkfs/work/workspace/scratch/lmu_chd4938-MINED_26/Qwen2.5-VL-null-space/LLaMA-Factory


# llamafactory-cli export /hkfs/work/workspace/scratch/lmu_chd4938-MINED_26/Qwen2.5-VL-null-space/LLaMA-Factory/examples/merge_lora/qwen2_5vl_lora_sft.yaml

llamafactory-cli train /hkfs/work/workspace/scratch/lmu_chd4938-MINED_26/Qwen2.5-VL-null-space/LLaMA-Factory/examples/train_lora/qwen2_5vl_lora_sft.yaml