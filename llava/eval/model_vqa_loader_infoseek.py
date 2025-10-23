import argparse
from audioop import lin2lin
import torch
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
import shortuuid
import sys
sys.path.append("/home/bingxing2/ailab/scx6mh7/jkl/LLaVA_8_8_null_space")
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader



from PIL import Image
import math
import numpy as np


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]

        
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')

        print('!!!!!!!!!!!!!!!!!!!!!!!',self.image_processor)
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')


        return input_ids, image_tensor, image.size



    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes



# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def calculate_perplexity(model, input_ids, image_tensor, image_sizes, tokenizer, cur_prompt, answer):
    """计算perplexity"""
    model.eval()
    
    prompt_ids = tokenizer(cur_prompt, return_tensors="pt")["input_ids"].to(device='cuda', non_blocking=True)
    prompt_len = prompt_ids.shape[-1] 
    full_text = cur_prompt + answer
    
    # 重新生成了 input_ids，它现在在 CPU 上
    input_ids = tokenizer(full_text, return_tensors="pt", padding=True, truncation=True)["input_ids"]
    # 需要把它移动到 GPU
    input_ids = input_ids.to(device='cuda', non_blocking=True)

    with torch.no_grad():
        # 现在所有输入都在同一个 CUDA 设备上了
        outputs = model(
            input_ids=input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            return_dict=True
        )
        logits = outputs.logits

        # ... (后续代码不变)
        shift_logits = logits[..., prompt_len - 1: -1, :].contiguous()
        shift_labels = input_ids[..., prompt_len:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        neg_log_likelihood = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        perplexity = torch.exp(neg_log_likelihood)
        perplexity = perplexity.item()

        print("Perplexity:", perplexity)
        print("Logits max:", logits.max().item())
        print("Logits min:", logits.min().item())
        
    return perplexity


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)
    
    # 存储所有perplexity值
    all_perplexities = []

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]
        
        # 移动来自 data_loader 的张量
        # 注意：这里的 input_ids 会在下面被覆盖，所以移不移都可以，但移动 image_sizes 是必须的
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        image_tensor = image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)
        # image_sizes = image_sizes.to(device='cuda', non_blocking=True) # <<<<<<<<<<<<<<<<< 1. 添加此行

        has_answer = "answer" in line
        
        if has_answer:
            answer = line["answer"]
            perplexity = calculate_perplexity(model, input_ids, image_tensor, image_sizes, tokenizer, cur_prompt, answer)
        else:
            perplexity = None  # 或者设置一个默认值
 


        all_perplexities.append(perplexity)
        
        print(f"Question ID: {idx}, Perplexity: {perplexity:.4f}")

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "perplexity": perplexity,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    
    # 打印所有数据的perplexity统计信息
    all_perplexities = np.array(all_perplexities)
    print(f"\n=== Perplexity Statistics ===")
    print(f"Total samples: {len(all_perplexities)}")
    print(f"Mean perplexity: {np.mean(all_perplexities):.4f}")
    print(f"Std perplexity: {np.std(all_perplexities):.4f}")
    print(f"Min perplexity: {np.min(all_perplexities):.4f}")
    print(f"Max perplexity: {np.max(all_perplexities):.4f}")
    print(f"Median perplexity: {np.median(all_perplexities):.4f}")
    
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)