import os
import numpy as np
import torch
from datasets import load_dataset
import random
import io
import json
import sys
import pandas as pd
from typing import List, Dict
import re



def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def sample_train_loaders(name, tokenizer, nsamples=128, seed=0, seqlen=2048):
    set_seed(seed)
    if "wikitext2" in name:
        traindata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="train",
        )
        traindata = "\n\n".join(traindata["text"])
    elif "c4" in name:
        traindata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        traindata = "\n\n".join(traindata["text"])
    else:
        raise NotImplementedError

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, len(traindata) - seqlen * 2 - 1)
        j = i + seqlen * 2
        # breakpoint()
        trainenc = tokenizer(traindata[i:j], return_tensors="pt")
        inp = trainenc.input_ids[:, :seqlen]
        trainloader.append(inp)
    return trainloader


def get_redpajama_train(tokenizer, percent=10, seed=3, batch_size=128, max_length=2048):
    def tokenization(example):
        return tokenizer(example["text"], truncation=True, max_length=max_length)

    if percent != 100:
        split = f"train[:{int(850000*percent/100)}]"
    else:
        split = "train"
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split=split)

    processed_dataset = dataset.map(
        tokenization, batched=True, batch_size=batch_size, num_proc=os.cpu_count()
    )
    return processed_dataset


def get_english_quote(dataset_name, tokenizer):
    data = load_dataset(dataset_name)
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
    return data["train"]


def get_qat_dataset(name, tokenizer, data_percent):
    if name == "red_pajama":
        data = get_redpajama_train(tokenizer, data_percent)

    elif name == "Abirate/english_quotes":
        data = get_english_quote(name, tokenizer)
    else:
        raise NotImplementedError
    data = data.shuffle()
    return data

'''
llama_chat_format="""<s>[INST] <<SYS>>
"Below is an instruction that describes a task. Write a response that appropriately completes the request."
<</SYS>>

{{ instruction }} [/INST] {{ response }} </s>
"""
'''

llama_chat_format="""<s>[INST] <<SYS>>
"Below is an instruction that describes a task. Write a response that appropriately completes the request."
<</SYS>>

{instruction} [/INST] {response} </s>
"""


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
        #f = open(f)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def get_calib_data(name, tokenizer, model_id, nsamples, seqlen=2048, seed=3):
    print(f" get_data_from: {name}, nsamples={nsamples}, seqlen={seqlen}, {seed}")
    cache_file = (
        f"cache/{name}_{model_id.replace('/','_')}_{nsamples}_{seqlen}_{seed}.pt"
    )
    random.seed(seed)
    if not os.path.exists("cache"):
        os.makedirs("cache")
    if os.path.exists(cache_file):
        print(f"found data file: {cache_file}")
        traindataset = torch.load(cache_file)
        print("loaded ...")
        return traindataset
    if name == "c4":
        traindata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        tot_text = "\n\n".join(traindata["text"])
    elif name == "wikitext2":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        tot_text = "\n\n".join(traindata["text"])
    elif name=="ptb":
        traindata = load_dataset(
            "ptb_text_only",
            "penn_treebank",
            split="train",
        )
        tot_text = "\n\n".join(traindata["sentence"])
    elif name == "traivia_qa":
        traindata = load_dataset("trivia_qa", "rc", split="train")
        tot_text = "\n\n".join(traindata["question"])
    elif name == "nqopen":
        traindata = load_dataset("nq_open", split="train")
        tot_text = "\n\n".join(traindata["question"])        
    elif name == "alpaca":
        # this is for chat models
        data_path="data/alpaca_data.json"
        list_data_dict = jload(data_path)
        traindataset =[]
        selected_data_dict=random.sample(list_data_dict, nsamples)
        #random_indices = np.random.choice(len(list_data_dict), nsamples, replace=False)
        #selected_data_dict = [list_data_dict[i] for i in random_indices]
        for example in selected_data_dict:
            if example.get("input", "") == "":
                s=llama_chat_format.format(instruction=example["instruction"], response=example["output"])
                trainenc=tokenizer(s, return_tensors="pt")
                inp=trainenc.input_ids[:, :seqlen]
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        print("example instruction:", s)
        torch.save(traindataset, cache_file)
        return traindataset
    elif name == "MetaMATH":
        data_path="data/MetaMathQA-395K.json"
        list_data_dict = jload(data_path)
        traindataset =[]
        selected_data_dict=random.sample(list_data_dict, nsamples)
        for example in selected_data_dict:
            if example.get("input", "") == "":
                s=llama_chat_format.format(instruction=example["query"], response=example["response"])
                trainenc=tokenizer(s, return_tensors="pt")
                inp=trainenc.input_ids[:, :seqlen]
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        print("example instruction:", s)        
        torch.save(traindataset, cache_file)
        return traindataset
    elif name == "codefeedback":
        data_path="data/CodeFeedback-Filtered-Instruction.jsonl"
        with open(data_path, 'r') as json_file:
            json_list = list(json_file)
        print(len(json_list))
        list_data_dict = []
        for item in json_list:
            dict_item = json.loads(item)
            list_data_dict.append(dict_item)
            assert isinstance(dict_item, dict)
        #list_data_dict = jload(data_path)
        traindataset =[]
        #selected_data_dict=random.sample(list_data_dict, nsamples)
        random_indices = np.random.choice(len(list_data_dict), nsamples, replace=False)
        selected_data_dict = [list_data_dict[i] for i in random_indices]        
        for example in selected_data_dict:
            if example.get("input", "") == "":
                s=llama_chat_format.format(instruction=example["query"], response=example["answer"])
                trainenc=tokenizer(s, return_tensors="pt")
                inp=trainenc.input_ids[:, :seqlen]
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        print("example instruction:", s) 
        torch.save(traindataset, cache_file)
        return traindataset
    elif name == "WizLMinstruct":
        data_path="data/WizardLM_evol_instruct_V2_143k.jsonl"
        with open(data_path, 'r') as json_file:
            json_list = list(json_file)
        print(len(json_list))
        list_data_dict = []
        for item in json_list:
            dict_item = json.loads(item)
            list_data_dict.append(dict_item)
            assert isinstance(dict_item, dict)
        #list_data_dict = jload(data_path)
        traindataset =[]
        selected_data_dict=random.sample(list_data_dict, nsamples)
        for example in selected_data_dict:
            if example.get("input", "") == "":
                s=llama_chat_format.format(instruction=example["conversation"][0]["human"], response=example["conversation"][0]["assistant"])
                trainenc=tokenizer(s, return_tensors="pt")
                inp=trainenc.input_ids[:, :seqlen]
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        print("example instruction:", s)        
        torch.save(traindataset, cache_file)
        return traindataset        
    else:
        raise NotImplementedError
    print(f"tot_text={len(tot_text)}")
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        inp = trainenc.input_ids[:, :seqlen]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    torch.save(traindataset, cache_file)
    return traindataset


def get_eval_loaders(name, tokenizer):
    if "wikitext2" in name:
        testdata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    if "ptb" in name:
        valdata = load_dataset(
            "ptb_text_only",
            "penn_treebank",
            split="validation",
        )
        testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")
        return testenc
    if "c4" in name:
        testdata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc        
    raise NotImplementedError

############################################################################
import json
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from torch.utils.data import Dataset, DataLoader
import os

from PIL import Image
import math

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

# 创建自定义数据集类
class MMEDataset(Dataset):
    def __init__(self, data_root, questions):
        self.image_folder = data_root
        self.questions = questions

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        line = self.questions[idx]
        qs = line["text"]
        # 加载图像
        image_file = line["image"]
        image_path = os.path.join(self.image_folder, image_file)
        image = Image.open(image_path).convert('RGB')
        # 调试输出：当为 OneVision 数据（包含 data_source 字段）时打印信息
        if 'data_source' in line:
            print('----------------使用OneVision数据--------------------------')
            print('data_source:', line['data_source'])
            print('image_file:', image_file)
            print('qs:', qs)
        return {"image": image, "question": qs, "image_path":image_path}

from qwen_vl_utils import process_vision_info

class DataCollator:
    def __init__(self, processor):
        self.processor = processor
        
    def __call__(self, examples):

        messages = []
        for example in examples:
            msg = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example['image_path']},
                        {"type": "text", "text": example['question']},
                    ],
                }
            ]
            messages.append(msg)
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs

# =========================
# 额外的通用数据加载工具（来自 llava/datautils.py 的思路）
# - load_benchmark_datasets: 通过 VLM-Eval 的统一入口采样图像与问题
# - load_onevision_from_local: 从本地 HuggingFace Dataset 目录按4个source分别采样并保存图片
# =========================

sys.path.append("/hkfs/work/workspace/scratch/lmu_chd4938-MINED_26/Qwen2.5-VL-null-space")
# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from llava.conversation import conv_templates, SeparatorStyle
# from llava.model.builder import load_pretrained_model
# from llava.utils import disable_torch_init
# from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
import math

from benchmark_load import process_vlmeval_datasets, DATASET_CONFIG


from benchmark_load import process_vlmeval_datasets, DATASET_CONFIG

def _clean_image_placeholders(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s*<image\s*\d*\s*>\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*<ImageHere\s*\d*\s*>\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_benchmark_datasets(dataset_names: List[str], n_samples_per_dataset: int,
                          data_root: str,
                          seed: int = 233, save_sampled_data: bool = True,
                          output_file: str = "sampled_benchmark_data.jsonl") -> List[Dict]:
    set_seed(seed)
    print(f"使用随机种子: {seed}")
    print(f"正在加载数据集: {dataset_names}")
    print(f"每个数据集采样 {n_samples_per_dataset} 个样本")
    dataset_results = process_vlmeval_datasets(dataset_names, data_root)
    all_sampled_data = []
    all_original_data = []

    for dataset_name, dataset_result in dataset_results.items():
        if dataset_result is None:
            print(f"⚠ 数据集 {dataset_name} 处理失败，跳过")
            continue
        print(f"处理数据集: {dataset_name}")
        print(f"  总样本数: {dataset_result['total_samples']}")
        print(f"  图像数: {dataset_result['image_count']}")
        data = dataset_result['data']
        image_paths = dataset_result['image_paths']
        valid_indices = []
        for idx, row in data.iterrows():
            has_question = not pd.isna(row.get('question', '')) and str(row.get('question', '')).strip() != ''
            has_image = str(row['index']) in image_paths
            if has_question and has_image:
                valid_indices.append(idx)
        print(f"  有效样本数（同时包含图像和问题）: {len(valid_indices)}")
        if len(valid_indices) == 0:
            print(f"⚠ 数据集 {dataset_name} 没有有效样本（同时包含图像和问题），跳过")
            continue
        if n_samples_per_dataset < len(valid_indices):
            sampled_valid_indices = random.sample(valid_indices, n_samples_per_dataset)
        else:
            sampled_valid_indices = valid_indices
            print(f"⚠ 请求的样本数 {n_samples_per_dataset} 大于数据集 {dataset_name} 的有效样本数 {len(valid_indices)}，使用所有有效样本")

        dataset_sampled_data = []
        for idx in sampled_valid_indices:
            row = data.iloc[idx]
            try:
                original_index = int(row['index'])
            except (ValueError, TypeError):
                original_index = str(row['index'])
            sample = {
                'dataset_name': dataset_name,
                'image': image_paths[str(row['index'])],
                'text': row.get('question', ''),
                'original_index': original_index,
                'sampled_index': len(dataset_sampled_data)
            }
            dataset_sampled_data.append(sample)
            all_sampled_data.append(sample)

        dataset_original_data = []
        for idx in valid_indices:
            row = data.iloc[idx]
            try:
                original_index = int(row['index'])
            except (ValueError, TypeError):
                original_index = str(row['index'])
            original_sample = {
                'dataset_name': dataset_name,
                'image': image_paths[str(row['index'])],
                'text': row.get('question', ''),
                'original_index': original_index,
                'is_sampled': idx in sampled_valid_indices
            }
            dataset_original_data.append(original_sample)
        all_original_data.extend(dataset_original_data)
        print(f"  ✓ 从数据集 {dataset_name} 采样了 {len(dataset_sampled_data)} 个样本")

    print(f"✓ 总共采样了 {len(all_sampled_data)} 个样本")

    if save_sampled_data:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in all_sampled_data:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            original_output_file = output_file.replace('.jsonl', '_original_complete.jsonl')
            with open(original_output_file, 'w', encoding='utf-8') as f:
                for sample in all_original_data:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            print(f"✓ 采样数据已保存到: {output_file}")
            print(f"✓ 完整原始数据已保存到: {original_output_file}")
            print(f"  - 采样数据包含 {len(all_sampled_data)} 个样本")
            print(f"  - 完整数据包含 {len(all_original_data)} 个样本")
        except Exception as e:
            print(f"⚠ 保存数据时出错: {e}")
    return all_sampled_data

def _save_pil_image(pil_img, path: str) -> bool:
    try:
        pil_img.save(path)
        return True
    except Exception as e:
        print(f"⚠ 保存图像失败: {path}, 错误: {e}")
        return False

def load_onevision_from_local(
    dataset_local_path: str,
    sources: List[str],
    n_samples_per_source: int,
    image_save_dir: str,
    output_file: str,
    seed: int = 233,
) -> List[Dict]:
    assert os.path.isdir(dataset_local_path), f"本地数据集目录不存在: {dataset_local_path}"
    assert isinstance(sources, list) and len(sources) == 4, "需要提供4个 source"
    os.makedirs(image_save_dir, exist_ok=True)
    set_seed(seed)

    print(f"从本地路径加载 OneVision: {dataset_local_path}")
    print(f"指定的 sources: {sources}")

    all_samples: List[Dict] = []
    for src in sources:
        print(f"加载 OneVision 子集 config='{src}'")
        try:
            ds = load_dataset(dataset_local_path, src, split="train")
            total = len(ds)
            print(f"  - 子集 '{src}' 总样本数: {total}")
            if total == 0:
                print(f"⚠ 子集 '{src}' 没有样本，跳过")
                continue
            if n_samples_per_source < total:
                indices = random.sample(range(total), n_samples_per_source)
            else:
                indices = list(range(total))
                print(f"⚠ 请求样本数 {n_samples_per_source} > 可用样本数 {total}，使用全部")

            for i in indices:
                ex = ds[i]
                ex_id = str(ex.get("id", i))
                pil_or_path = ex.get("image", None)
                if pil_or_path is None:
                    continue

                text = ""
                convs = ex.get("conversations", []) or []
                if isinstance(convs, list):
                    for m in convs:
                        if isinstance(m, dict) and m.get("from") == "human":
                            text = _clean_image_placeholders(str(m.get("value", "")).strip())
                            if text:
                                break
                if not text:
                    continue

                img_dst = os.path.join(image_save_dir, f"{ex_id}.jpg")
                ok = False
                if isinstance(pil_or_path, Image.Image):
                    ok = _save_pil_image(pil_or_path.convert("RGB"), img_dst)
                elif isinstance(pil_or_path, str) and os.path.exists(pil_or_path):
                    try:
                        im = Image.open(pil_or_path).convert("RGB")
                        ok = _save_pil_image(im, img_dst)
                    except Exception as e:
                        print(f"⚠ 打开原图失败: {pil_or_path}, 错误: {e}")
                        ok = False
                else:
                    try:
                        import numpy as np
                        if hasattr(pil_or_path, "__array_interface__"):
                            arr = np.asarray(pil_or_path)
                            im = Image.fromarray(arr).convert("RGB")
                            ok = _save_pil_image(im, img_dst)
                    except Exception:
                        ok = False
                if not ok or not os.path.exists(img_dst):
                    continue

                all_samples.append({
                    "dataset_name": "ONEVISION",
                    "data_source": src,
                    "image": os.path.abspath(img_dst),
                    "text": text,
                    "original_index": ex_id
                })

            print(f"  ✓ 从子集 '{src}' 成功处理 {len([s for s in all_samples if s['data_source'] == src])} 个样本")

        except Exception as e:
            print(f"⚠ 加载子集失败: {src}，错误: {e}")
            continue

    print(f"✓ 总共处理了 {len(all_samples)} 个样本")

    if output_file:
        try:
            out_dir = output_file.rsplit("/", 1)[0] if "/" in output_file else "."
            os.makedirs(out_dir, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                for r in all_samples:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"✓ 已保存合并样本到: {output_file} (共 {len(all_samples)} 条)")
        except Exception as e:
            print(f"⚠ 保存样本索引失败: {e}")

    return all_samples

def get_calib_data_mllm(args):
    # Model
    model_path = args.model_id
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    model = model.eval()
    processor = AutoProcessor.from_pretrained(model_path)

    # 根据参数来源选择数据构建 questions 列表
    if hasattr(args, 'dataset_names') and args.dataset_names is not None:
        lower_names = [str(x).lower() for x in args.dataset_names]
        if 'onevision' in lower_names:
            print("使用 OneVision 本地数据集")
            assert hasattr(args, 'onevision_sources') and isinstance(args.onevision_sources, list) and len(args.onevision_sources) == 4, "需要 args.onevision_sources 为包含4个元素的list"
            assert hasattr(args, 'onevision_local_dir') and args.onevision_local_dir, "需要提供 --onevision_local_dir 作为本地数据集路径"
            assert hasattr(args, 'onevision_image_dir') and args.onevision_image_dir, "需要提供 --onevision_image_dir 作为图片保存目录"

            n_per = int(args.n_samples_per_dataset)
            output_file = getattr(args, 'output_file', f"sampled_onevision_seed{args.seed}.jsonl")

            os.makedirs(args.onevision_image_dir, exist_ok=True)

            questions = load_onevision_from_local(
                dataset_local_path=args.onevision_local_dir,
                sources=args.onevision_sources,
                n_samples_per_source=n_per,
                image_save_dir=args.onevision_image_dir,
                output_file=output_file,
                seed=args.seed,
            )
        else:
            # 使用 benchmark 数据集
            print(f"使用benchmark数据集: {args.dataset_names}")
            print(f"每个数据集采样 {args.n_samples_per_dataset} 个样本")
            output_file = getattr(args, 'output_file', f"sampled_benchmark_seed{args.seed}.jsonl")
            output_dir = output_file.rsplit('/', 1)[0] if '/' in output_file else "."
            os.makedirs(output_dir, exist_ok=True)
            print(f"✓ 创建输出目录: {output_dir}")
            print(f"✓ 输出文件路径: {output_file}")
            questions = load_benchmark_datasets(
                dataset_names=args.dataset_names,
                n_samples_per_dataset=int(args.n_samples_per_dataset),
                data_root=getattr(args, 'data_root', "/home/jiangkailin/mydisk/iclr26_evoke_dynamic_null_space/cache/vlmeval"),
                seed=args.seed,
                save_sampled_data=True,
                output_file=output_file,
            )
    else:
        # 旧的 question_file 模式
        questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
        questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    print("Creating data loader...")
    # 这里 image_folder 可留空，因为 questions 中 image 多为绝对路径
    dataset = MMEDataset(getattr(args, 'image_folder', ''), questions)
    collater = DataCollator(processor)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collater)
    return model, processor, data_loader