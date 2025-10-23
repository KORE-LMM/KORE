import subprocess

def main():
    cmd = [
        "python", "-m", "llava.eval.model_vqa_loader",
        "--model-path", "/media/raid/workspace/jiangkailin/LLaVA/mllm_save_LoRA_Null_adapter_llama2_PT_128_math_Null_v1_merged",
        "--question-file", "/media/raid/workspace/jiangkailin/LLaVA/playground/data/eval/MME/llava_mme.jsonl",
        "--image-folder", "/media/raid/workspace/jiangkailin/LLaVA/playground/data/eval/MME/MME_Benchmark_release_version/MME_Benchmark",
        "--answers-file", "./playground/data/eval/MME/answers/save_LoRA_Null_adapter_llama2_PT_128_math_Null_v1_merged.jsonl",
        "--temperature", "0",
        "--conv-mode", "vicuna_v1"
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()