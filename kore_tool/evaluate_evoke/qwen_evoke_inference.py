import json
import os
import torch
import multiprocessing as mp
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import uuid
import argparse
from typing import List, Dict, Any
import threading
import time

def load_model_on_gpu(gpu_id: int):
    """Load model on the specified GPU."""
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    
    print(f"Loading model on GPU {gpu_id}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "/home/bingxing2/ailab/group/ai4bio/renyuchen/jkl/qwen_vl_train_ckpt/plain_lora_Qwen2.5-VL-7B-Instruct/nwe_merge_plain_lora_Qwen2.5-VL-7B-Instruct", 
        torch_dtype="auto", 
        device_map={"": device}
    )
    
    processor = AutoProcessor.from_pretrained("/home/bingxing2/ailab/group/ai4bio/renyuchen/jkl/qwen_vl_train_ckpt/plain_lora_Qwen2.5-VL-7B-Instruct/nwe_merge_plain_lora_Qwen2.5-VL-7B-Instruct")
    
    return model, processor
def process_batch(gpu_id: int, data_batch: List[Dict], output_file: str, image_prefix: str = ""):
    """Process one batch of data."""
    try:
        # Load model
        model, processor = load_model_on_gpu(gpu_id)
        
        # Create a file lock to ensure safe multi-process writing
        file_lock = threading.Lock()
        
        for item in tqdm(data_batch, desc=f"GPU {gpu_id}"):
            try:
                # Build the full image path
                image_path = item["image"]
                if image_prefix:
                    # Ensure the prefix ends with '/' and the image path does not start with '/'
                    if not image_prefix.endswith('/'):
                        image_prefix += '/'
                    if image_path.startswith('/'):
                        image_path = image_path[1:]
                    full_image_path = image_prefix + image_path
                else:
                    full_image_path = image_path
                
                # Build messages
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": full_image_path,
                            },
                            {"type": "text", "text": item["text"]},
                        ],
                    }
                ]
                
                # Prepare inputs for inference
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(model.device)
                
                # Run generation
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=128)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                
                # Build result
                result = {
                    "question_id": item["question_id"],
                    "prompt": item["text"],
                    "text": output_text[0].strip(),
                    "answer_id": str(uuid.uuid4()),
                    "model_id": "Qwen2.5-VL-7B-Instruct",
                    "metadata": {}
                }
                
                # Write result to file immediately
                with file_lock:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f.flush()  # Force flush to ensure immediate write to disk
                
                print(f"GPU {gpu_id}: finished question_id {item['question_id']}")
                
            except Exception as e:
                print(f"Error processing item {item['question_id']} on GPU {gpu_id}: {e}")
                # Add error result
                result = {
                    "question_id": item["question_id"],
                    "prompt": item["text"],
                    "text": "ERROR",
                    "answer_id": str(uuid.uuid4()),
                    "model_id": "Qwen2.5-VL-7B-Instruct",
                    "metadata": {"error": str(e)}
                }
                
                # Write error result immediately
                with file_lock:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f.flush()
                
                print(f"GPU {gpu_id}: error on question_id {item['question_id']}")
        
        print(f"GPU {gpu_id} completed processing {len(data_batch)} items.")
        
    except Exception as e:
        print(f"Error on GPU {gpu_id}: {e}")
        # Save error information
        file_lock = threading.Lock()
        for item in data_batch:
            error_result = {
                "question_id": item["question_id"],
                "prompt": item["text"],
                "text": "ERROR",
                "answer_id": str(uuid.uuid4()),
                "model_id": "Qwen2.5-VL-7B-Instruct",
                "metadata": {"error": str(e)}
            }
            
            with file_lock:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(error_result, ensure_ascii=False) + '\n')
                    f.flush()

def load_jsonl_data(file_path: str) -> List[Dict]:
    """Load JSONL data."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def split_data(data: List[Dict], num_gpus: int) -> List[List[Dict]]:
    """Split data into multiple batches."""
    batch_size = len(data) // num_gpus
    batches = []
    
    for i in range(num_gpus):
        start_idx = i * batch_size
        if i == num_gpus - 1:  # Let the last GPU handle all remaining data
            end_idx = len(data)
        else:
            end_idx = (i + 1) * batch_size
        
        batches.append(data[start_idx:end_idx])
    
    return batches

def merge_results(output_files: List[str], final_output: str):
    """Merge results from all GPUs."""
    all_results = []
    
    for file_path in output_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        all_results.append(json.loads(line.strip()))
    
    # Sort by question_id
    all_results.sort(key=lambda x: x['question_id'])
    
    # Save final merged results
    with open(final_output, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Merging finished, processed {len(all_results)} items in total.")

def monitor_progress(output_files: List[str], total_items: int):
    """Monitor progress."""
    processed_count = 0
    while processed_count < total_items:
        time.sleep(10)  # Check every 10 seconds
        current_count = 0
        for file_path in output_files:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    current_count += sum(1 for line in f if line.strip())
        
        if current_count > processed_count:
            print(f"Progress: {current_count}/{total_items} ({current_count/total_items*100:.1f}%)")
            processed_count = current_count

def main():
    parser = argparse.ArgumentParser(description='Multi-GPU inference for Qwen2.5-VL model.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input JSONL file.')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save outputs.')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use.')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct', help='Model name.')
    parser.add_argument('--image_prefix', type=str, default='', help='Image path prefix, e.g. /path/to/images/')
    parser.add_argument('--monitor', action='store_true', help='Enable progress monitoring.')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    data = load_jsonl_data(args.input_file)
    print(f"Loaded {len(data)} items in total.")
    
    # Split data
    data_batches = split_data(data, args.num_gpus)
    print(f"Data has been split into {len(data_batches)} batches.")
    for i, batch in enumerate(data_batches):
        print(f"GPU {i}: {len(batch)} items")
    
    # Prepare output file paths
    output_files = []
    for i in range(args.num_gpus):
        output_file = os.path.join(args.output_dir, f'gpu_{i}_results.jsonl')
        output_files.append(output_file)
        # Truncate files to start from clean state
        with open(output_file, 'w', encoding='utf-8') as f:
            pass
    
    # Start monitoring thread
    monitor_thread = None
    if args.monitor:
        monitor_thread = threading.Thread(
            target=monitor_progress, 
            args=(output_files, len(data))
        )
        monitor_thread.daemon = True
        monitor_thread.start()
    
    # Launch multi-process inference
    processes = []
    for i in range(args.num_gpus):
        if len(data_batches[i]) > 0:  # Only process non-empty batches
            p = mp.Process(target=process_batch, args=(i, data_batches[i], output_files[i], args.image_prefix))
            p.start()
            processes.append(p)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Wait for monitor thread to finish
    if monitor_thread:
        monitor_thread.join(timeout=1)
    
    # Merge results
    final_output = os.path.join(args.output_dir, 'final_results.jsonl')
    merge_results(output_files, final_output)
    
    print("Inference done!")

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()