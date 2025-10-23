import os
import numpy as np
import torch
from datasets import load_dataset
import random
import io
import json
import sys
import pandas as pd
from typing import List, Dict, Optional
from tqdm import tqdm
import glob
import re
from PIL import Image

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

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

llama_chat_format="""<s>[INST] <<SYS>>
"Below is an instruction that describes a task. Write a response that appropriately completes the request."
<</SYS>>

{instruction} [/INST] {response} </s>
"""

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def get_calib_data(name, tokenizer, model_id, nsamples, seqlen=2048, seed=3, question_file=None):
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
        data_path="data/alpaca_data.json"
        list_data_dict = jload(data_path)
        traindataset =[]
        selected_data_dict=random.sample(list_data_dict, nsamples)
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
        list_data_dict = []
        for item in json_list:
            dict_item = json.loads(item)
            list_data_dict.append(dict_item)
            assert isinstance(dict_item, dict)
        traindataset =[]
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
        list_data_dict = []
        for item in json_list:
            dict_item = json.loads(item)
            list_data_dict.append(dict_item)
            assert isinstance(dict_item, dict)
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

########################################LLaVA load benchmark data #####################################################

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
import math

from benchmark_load import process_vlmeval_datasets, DATASET_CONFIG

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def load_benchmark_datasets(dataset_names: List[str], n_samples_per_dataset: int, 
                          data_root: str = "cache/vlmeval",
                          seed: int = 233, save_sampled_data: bool = True, 
                          output_file: str = "sampled_benchmark_data.jsonl") -> List[Dict]:

    set_seed(seed)
    print(f"Using random seed: {seed}")
    print(f"Loading datasets: {dataset_names}")
    print(f"Sampling {n_samples_per_dataset} samples per dataset")
    dataset_results = process_vlmeval_datasets(dataset_names, data_root)
    all_sampled_data = []
    all_original_data = []
    for dataset_name, dataset_result in dataset_results.items():
        if dataset_result is None:
            print(f"⚠ Dataset {dataset_name} processing failed, skipping")
            continue
        print(f"Processing dataset: {dataset_name}")
        print(f"  Total samples: {dataset_result['total_samples']}")
        print(f"  Image count: {dataset_result['image_count']}")
        data = dataset_result['data']
        image_paths = dataset_result['image_paths']
        valid_indices = []
        for idx, row in data.iterrows():
            has_question = not pd.isna(row.get('question', '')) and str(row.get('question', '')).strip() != ''
            has_image = str(row['index']) in image_paths
            if has_question and has_image:
                valid_indices.append(idx)
        print(f"  Valid samples (with both image and question): {len(valid_indices)}")
        if len(valid_indices) == 0:
            print(f"⚠ Dataset {dataset_name} has no valid samples (with both image and question), skipping")
            continue
        if n_samples_per_dataset < len(valid_indices):
            sampled_valid_indices = random.sample(valid_indices, n_samples_per_dataset)
        else:
            sampled_valid_indices = valid_indices
            print(f"⚠ Requested sample count {n_samples_per_dataset} is greater than valid samples {len(valid_indices)} in dataset {dataset_name}, using all valid samples")

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
        print(f"  ✓ Sampled {len(dataset_sampled_data)} samples from dataset {dataset_name}")
    print(f"✓ Total sampled {len(all_sampled_data)} samples")

    if save_sampled_data:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in all_sampled_data:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            original_output_file = output_file.replace('.jsonl', '_original_complete.jsonl')
            with open(original_output_file, 'w', encoding='utf-8') as f:
                for sample in all_original_data:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            print(f"✓ Sampled data saved to: {output_file}")
            print(f"✓ Complete original data saved to: {original_output_file}")
            print(f"  - Sampled data contains {len(all_sampled_data)} samples")
            print(f"  - Complete data contains {len(all_original_data)} samples")
        except Exception as e:
            print(f"⚠ Error saving data: {e}")
    return all_sampled_data

######################## OneVision: Local path + 4 sources + Save images ########################

def _clean_image_placeholders(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s*<image\s*\d*\s*>\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*<ImageHere\s*\d*\s*>\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _save_pil_image(pil_img, path: str) -> bool:
    try:
        pil_img.save(path)
        return True
    except Exception as e:
        print(f"⚠ Failed to save image: {path}, error: {e}")
        return False

def load_onevision_from_local(
    dataset_local_path: str,
    sources: List[str],
    n_samples_per_source: int,
    image_save_dir: str,
    output_file: str,
    seed: int = 233,
) -> List[Dict]:
    """
    Load OneVision dataset from local path, call load_dataset(<local_path>, <config>, split='train') for each source,
    randomly sample n_samples_per_source items from each source.
    Save images to image_save_dir/<id>.jpg; extract text from first human statement.
    Return merged list [{image: abs_path, text: str, ...}, ...].
    """
    assert os.path.isdir(dataset_local_path), f"Local dataset directory does not exist: {dataset_local_path}"
    assert isinstance(sources, list) and len(sources) == 4, "Need to provide 4 sources"
    os.makedirs(image_save_dir, exist_ok=True)
    set_seed(seed)

    print(f"Loading OneVision from local path: {dataset_local_path}")
    print(f"Specified sources: {sources}")

    all_samples: List[Dict] = []
    for src in sources:
        print(f"Loading OneVision subset config='{src}'")
        try:
            # Call load_dataset for each source separately, specifying config
            ds = load_dataset(dataset_local_path, src, split="train")
            total = len(ds)
            print(f"  - Subset '{src}' total samples: {total}")
            
            if total == 0:
                print(f"⚠ Subset '{src}' has no samples, skipping")
                continue

            if n_samples_per_source < total:
                indices = random.sample(range(total), n_samples_per_source)
            else:
                indices = list(range(total))
                print(f"⚠ Requested sample count {n_samples_per_source} > available samples {total}, using all")

            for i in indices:
                ex = ds[i]
                ex_id = str(ex.get("id", i))
                pil_or_path = ex.get("image", None)
                if pil_or_path is None:
                    continue

                # Text: extract first human statement
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

                # Save image
                img_dst = os.path.join(image_save_dir, f"{ex_id}.jpg")
                ok = False
                if isinstance(pil_or_path, Image.Image):
                    ok = _save_pil_image(pil_or_path.convert("RGB"), img_dst)
                elif isinstance(pil_or_path, str) and os.path.exists(pil_or_path):
                    try:
                        im = Image.open(pil_or_path).convert("RGB")
                        ok = _save_pil_image(im, img_dst)
                    except Exception as e:
                        print(f"⚠ Failed to open original image: {pil_or_path}, error: {e}")
                        ok = False
                else:
                    # Might be array-like
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
            
            print(f"  ✓ Successfully processed {len([s for s in all_samples if s['data_source'] == src])} samples from subset '{src}'")
            
        except Exception as e:
            print(f"⚠ Failed to load subset: {src}, error: {e}")
            continue

    print(f"✓ Total processed {len(all_samples)} samples")

    # Optional: save index
    if output_file:
        try:
            out_dir = output_file.rsplit("/", 1)[0] if "/" in output_file else "."
            os.makedirs(out_dir, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                for r in all_samples:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"✓ Saved merged samples to: {output_file} (total {len(all_samples)} items)")
        except Exception as e:
            print(f"⚠ Failed to save sample index: {e}")

    return all_samples

# =========================
# Dataset / DataLoader
# =========================

class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode,output_file, n_samples=None):
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode
        self.output_file = output_file
        if n_samples is not None and n_samples < len(questions):
            self.questions = random.sample(questions, n_samples)
        else:
            self.questions = questions
    
    def __getitem__(self, index):
        line = self.questions[index]
        if self.output_file:
            try:
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    json.dump(line, f, ensure_ascii=False)
                    f.write('\n')
            except Exception as e:
                print(f"Error saving line data to jsonl file: {e}")
        
        if isinstance(line, dict) and 'image' in line:
            image_file = line['image']
            qs = line['text']
            print('----------------Using benchmark_load--------------------------')
            # Check if it's onevision data and print source info
            if 'data_source' in line:
                print('----------------Using OneVision data--------------------------')
                print('data_source:', line['data_source'])
                print('image_file:', image_file)
                print('qs:', qs)
            else:
                print('----------------Using benchmark_load--------------------------')
                print('image_file:', image_file)
                print('qs:', qs)

                
            if image_file and os.path.isabs(image_file):
                image_path = image_file
            elif image_file:
                image_path = os.path.join(self.image_folder, image_file)
            else:
                raise ValueError(f"Sample {index} has no image file")
        else:
            image_file = line["image"]
            qs = line["text"]
            image_path = os.path.join(self.image_folder, image_file)
        
        if not os.path.exists(image_path):
            raise ValueError(f"Image file does not exist: {image_path}")

        qs = _clean_image_placeholders(qs)

        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        image_size = image.size

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        
        return input_ids, image_tensor, image_size

    def __len__(self):
        return len(self.questions)

def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes

def create_data_loader_from_benchmark(dataset_names: List[str], n_samples_per_dataset: int, 
                                    tokenizer, image_processor, model_config, conv_mode,
                                    batch_size=1, num_workers=4, 
                                    data_root="/home/bingxing2/ailab/scx6mh7/jkl/ckpt_sum/data/vlmeval",
                                    seed=233, save_sampled_data=True, output_file="sampled_benchmark_data.jsonl"):
    assert batch_size == 1, "batch_size must be 1"
    questions = load_benchmark_datasets(dataset_names, n_samples_per_dataset, data_root, 
                                      seed, save_sampled_data, output_file)
    dataset = CustomDataset(questions, "", tokenizer, image_processor, model_config, conv_mode,output_file, n_samples=None)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, 
                           shuffle=False, collate_fn=collate_fn)
    return data_loader

def create_data_loader_from_onevision_local(
    dataset_local_path: str,
    sources: List[str],
    n_samples_per_source: int,
    image_save_dir: str,
    tokenizer, image_processor, model_config, conv_mode,
    batch_size=1, num_workers=4, seed=233,
    save_sampled_data=True,
    output_file="sampled_onevision_data.jsonl",
):
    """
    According to your new convention: load OneVision from local path, provide 4 sources, randomly sample n items from each, merge and return.
    Images saved to image_save_dir.
    """
    assert batch_size == 1, "batch_size must be 1"
    questions = load_onevision_from_local(
        dataset_local_path=dataset_local_path,
        sources=sources,
        n_samples_per_source=n_samples_per_source,
        image_save_dir=image_save_dir,
        output_file=output_file,
        seed=seed,
    )
    dataset = CustomDataset(questions, "", tokenizer, image_processor, model_config, conv_mode, output_file, n_samples=None)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader

def create_data_loader(nsamples, questions, image_folder, tokenizer, image_processor, model_config, conv_mode, output_file,batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, conv_mode, output_file,nsamples) 
    print('dataset',dataset)   
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader

def get_calib_data_mllm(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_id)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,
            device_map="auto", torch_dtype=torch.float32, trust_remote_code=True)
    model = model.to(dtype=torch.float32)
    
    if hasattr(args, 'dataset_names') and args.dataset_names is not None:
        if 'onevision' in [str(x).lower() for x in args.dataset_names]:
            print("Using OneVision local dataset")
            assert hasattr(args, 'onevision_sources') and isinstance(args.onevision_sources, list) and len(args.onevision_sources) == 4, "Need args.onevision_sources to be a list with 4 elements"
            assert hasattr(args, 'onevision_local_dir') and args.onevision_local_dir, "Need to provide --onevision_local_dir as local dataset path"
            assert hasattr(args, 'onevision_image_dir') and args.onevision_image_dir, "Need to provide --onevision_image_dir as image save directory"

            n_per = int(args.n_samples_per_dataset)
            output_file = getattr(args, 'output_file', f"sampled_onevision_seed{args.seed}.jsonl")

            os.makedirs(args.onevision_image_dir, exist_ok=True)

            data_loader = create_data_loader_from_onevision_local(
                dataset_local_path=args.onevision_local_dir,
                sources=args.onevision_sources,
                n_samples_per_source=n_per,
                image_save_dir=args.onevision_image_dir,
                tokenizer=tokenizer,
                image_processor=image_processor,
                model_config=model.config,
                conv_mode=args.conv_mode,
                seed=args.seed,
                save_sampled_data=True,
                output_file=output_file
            )
        else:
            # Original benchmark dataset processing
            print(f"Using benchmark dataset: {args.dataset_names}")
            print(f"Sampling {args.n_samples_per_dataset} samples per dataset")
            output_file = args.output_file
            output_dir = output_file.rsplit('/', 1)[0] if '/' in output_file else "."
            os.makedirs(output_dir, exist_ok=True)
            print(f"✓ Created output directory: {output_dir}")
            print(f"✓ Output file path: {output_file}")
            data_loader = create_data_loader_from_benchmark(
                args.dataset_names, 
                args.n_samples_per_dataset, 
                tokenizer, 
                image_processor, 
                model.config, 
                args.conv_mode,
                data_root=getattr(args, 'data_root', "/home/jiangkailin/mydisk/iclr26_evoke_dynamic_null_space/cache/vlmeval"),
                seed=args.seed,
                save_sampled_data=True,
                output_file=output_file
            )
    else:
        # Old ScienceQA / question_file mode
        questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
        questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
        if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
            args.conv_mode = args.conv_mode + '_mmtag'
            print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')
        data_loader = create_data_loader(args.calib_loader_size, questions, args.image_folder, tokenizer, image_processor, model.config, args.conv_mode, args.output_file)
    
    return model, data_loader