import numpy as np
import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
import os
import sys
import json
from pathlib import Path

# Add lora_null and qwen-vl-finetune paths to sys.path (for local custom classes)
sys.path.append('kore_qwen25_vl\lora_null')


from adapterlib.decomposition import CorDA_adapter
try:
    from model import CovSVDLinear, CovSVDQwen2_5_VLForConditionalGeneration
except Exception:
    CovSVDLinear = None
    CovSVDQwen2_5_VLForConditionalGeneration = None
from qwen_vl_utils import process_vision_info


def _maybe_sanitize_config(checkpoint_path: str) -> None:
    """If config.json contains fields like auto_map pointing to missing files,
    remove them to avoid loading failures.
    """
    cfg_path = Path(checkpoint_path) / 'config.json'
    if not cfg_path.exists():
        return
    try:
        cfg = json.load(open(cfg_path, 'r', encoding='utf-8'))
    except Exception:
        return

    changed = False
    # Clean legacy ours-llama config mappings to avoid missing configuration_oursvd_llama.py
    for k in ['auto_map', 'architectures']:
        if k in cfg and isinstance(cfg[k], (dict, list)):
            # If there is a custom mapping, remove it and let code decide the architecture
            if k == 'auto_map':
                changed = True
                del cfg['auto_map']
            elif k == 'architectures':
                # Let the script write back the correct architectures at the end
                changed = True
                del cfg['architectures']
    # Keep legacy fields such as lora_r for reference; no special handling

    if changed:
        backup = cfg_path.with_suffix('.json.bak')
        try:
            if not backup.exists():
                os.replace(str(cfg_path), str(backup))
        except Exception:
            pass
        json.dump(cfg, open(cfg_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)


def _load_model_any(model_id: str):
    # First try to fix bad configs (e.g. auto_map pointing to missing files)
    if os.path.isdir(model_id):
        _maybe_sanitize_config(model_id)

    # Prefer loading custom Null-SVD class if available
    if CovSVDQwen2_5_VLForConditionalGeneration is not None:
        try:
            model = CovSVDQwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map="auto",
            )
            return model
        except Exception:
            pass

    # Fallback to the official class (works even if adapters are already merged into the checkpoint)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model


def _is_adapter_like(module: nn.Module) -> bool:
    # Support both CorDA_adapter and CovSVDLinear (or other adapters with the same interface)
    return (
        hasattr(module, 'ALinear') and isinstance(getattr(module, 'ALinear', None), nn.Linear)
        and hasattr(module, 'BLinear') and isinstance(getattr(module, 'BLinear', None), nn.Linear)
        and hasattr(module, 'weight_residual')
    )


def main(args):
    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # More robust loading pipeline
    model = _load_model_any(model_id)


    print("\n---- model before merge ---\n")
    print(model)


    
    # Only operate on the LM part, similar to build_model2_mllm
    full_name_dict = {module: name for name, module in model.model.layers.named_modules()}
    linear_info = {}
    modules = [model.model.layers]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear) or _is_adapter_like(raw_linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)
    #print(linear_info)
    # ======= merge adapters into base model =======
    print("\nbegin merge. \n")
    # Only operate on the LM part
    module_dict = {module: name for name, module in model.model.layers.named_modules()}
    for module in module_dict.keys():
        name = module_dict[module]
        if _is_adapter_like(module):
            info = linear_info.get(module)
            if info is None:
                continue
            in_features = module.BLinear.in_features
            out_features = module.ALinear.out_features
            bias_flag = getattr(module.ALinear, 'bias', None) is not None

            with torch.no_grad():
                # Move weights to CPU and use memory-efficient dtype for computation
                a_w = module.ALinear.weight.detach().to('cpu', dtype=torch.float16)
                b_w = module.BLinear.weight.detach().to('cpu', dtype=torch.float16)
                res_w = module.weight_residual.detach().to('cpu', dtype=torch.float16)
                merged_weight = torch.matmul(a_w, b_w).add_(res_w).to(torch.float16)
                bias_cpu = None
                if bias_flag and getattr(module.ALinear, 'bias', None) is not None:
                    bias_cpu = module.ALinear.bias.detach().to('cpu', dtype=torch.float16)

                # Create a new Linear layer on CPU and assign merged weights to avoid GPU peak memory
                new_linear = nn.Linear(in_features, out_features, bias=bias_flag)
                new_linear.to('cpu')
                new_linear.weight.data = merged_weight.to(dtype=new_linear.weight.dtype)
                if bias_flag and bias_cpu is not None:
                    new_linear.bias.data = bias_cpu.to(dtype=new_linear.bias.dtype)

                # Replace the old adapter layer with the new Linear layer to free memory
                delattr(info["father"], info["name"])
                setattr(info["father"], info["name"], new_linear)
                # Proactively clean up CUDA memory fragments
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    print("\n---- model after merge ---\n")
    print(model)

    """
    # evaluate again:
    result = evaluate_model(
        model,
        tokenizer,
        args.model_id,
        "",
        eval_ppl="wikitext2,ptb",
        limit=-1,
    )
    print("Wiki PTB perplexity afater merge (used to check the difference before and after merging) ")
    print(result)
    """



    # Save as a Hugging Face model
    if args.save_model:
        assert args.save_path is not None
        save_path = args.save_path
        os.makedirs(save_path, exist_ok=True)

        tokenizer.save_pretrained(save_path)
        processor.save_pretrained(save_path)
        model.save_pretrained(save_path)
        config = model.config.to_dict()
        
        # Clean the config and remove adapter-related fields
        if "lora_r" in config:
            del config["lora_r"]
        if "auto_map" in config:
            del config["auto_map"]
        if "_name_or_path" in config:
            del config["_name_or_path"]
        
        # Ensure the architecture name is correct (use the pure official class)
        config["architectures"] = ["Qwen2_5_VLForConditionalGeneration"]
        
        import json
        json.dump(config, open(save_path + "/config.json", "w"), indent=2)

        print(f"Done merging adapter into the original model architecture in {save_path}")
        del model
        del tokenizer
        del processor
    # finished

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--save_model",
        default=True,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    main(args)