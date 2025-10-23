import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM
from adapterlib.datautils import get_calib_data, get_calib_data_mllm
from adapterlib.act_aware_utils import calib_input_distribution, calib_fisher_info, calib_cov_distribution,calib_infoseek_cov_distribution
from adapterlib.decomposition import build_model,build_model2,build_model2_mllm
import numpy as np
import os

def main(args):
    # setting random seed of numpy and torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # Load model
    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model, calib_loader = get_calib_data_mllm(args)
    
    # collect covariance for CO-SVD or activation for ASVD
    if args.act_aware:
        print('Collect activation-aware data for ASVD ...')
        if "fisher" in args.scaling_method:
            calib_fisher_info(model, calib_loader, args.use_cache)
        if "abs" in args.scaling_method:
            calib_input_distribution(
                model, calib_loader, args.scaling_method, args.use_cache
            )

    elif args.singular_aware:
        print('Collecting covariance data for Singular_aware ...')
        calib_cov_distribution(
            model, calib_loader, args.use_cache, args.calib_dataset, args.calib_loader_size, args.dataset_names, args.n_samples_per_dataset, args.benchmark_cache_file, seed=args.seed
        )

    elif args.singular_aware_2:
        print('Collecting covariance data for Singular_aware ...')
        calib_cov_distribution(
            model, calib_loader, args.use_cache, args.calib_dataset, args.calib_loader_size, seed=args.seed
        )
    else:
        print('Use the normal SVD ...')

    # perform decomposition
    if args.first_eigen:
        print("\n --- use the first r eigen vecs as adapters --- \n")
    else:
        print("\n --- use the last r eigen vecs as adapters --- \n")
    
    # build_model2(model, args)
    build_model2_mllm(model, args)
    
    ## save as hugging face model 
    if args.save_model:
        #assert args.cov_aware == True or args.singular_aware == True or args.singular_aware_2 == True
        assert args.save_path is not None
        save_path = args.save_path
        os.makedirs(save_path,exist_ok=True)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        config = model.config.to_dict()
        config["lora_r"] = args.r
        #config["atten_diag"] = args.atten_diag
        config["auto_map"] = {
            "AutoConfig": "configuration_oursvd_llama.CovSVDLlamaConfig",
            "AutoModelForCausalLM": "modeling_oursvd_llama.CovSVDLlamaForCausalLM",
        }
        config["architectures"] = ["CovSVDLlamaForCausalLM"]
        os.system(
            "cp ./mapping/configuration_oursvd_llama.py ./mapping/modeling_oursvd_llama.py ./"
            + save_path
        )
        import json

        json.dump(config, open(save_path + "/config.json", "w"), indent=2)

        print(f"Done building huggingface model in {save_path}")
        del model
        del tokenizer
    # finished

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--act_aware",
        action="store_true",
        help="use act aware svd (ASVD)",
    )
    parser.add_argument(
        "--cov_aware",
        action="store_true",
    )
    
    parser.add_argument("--singular_aware", 
                        action="store_true")
    parser.add_argument("--singular_aware_2", 
                        action="store_true")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="hyper-parameter alpha for ASVD",
    )
    parser.add_argument(
        "--calib_loader_size",
        type=int,
        default=256,
        help="number of samples used for covariance matrices",
    )    
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb", "traivia_qa", "nqopen", "MetaMATH", "codefeedback", "WizLMinstruct", "alpaca", "MME", "scienceqa"],
        help="calibration dataset",
    )
    parser.add_argument(
        "--scaling_method",
        type=str,
        default="abs_mean",
        choices=["abs_mean", "abs_max", "fisher", "fisher_abs_mean"],
        help="scaling method",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="use cached calibration results",
    )

    parser.add_argument(
        "--eval_mmlu",
        action="store_true",
        help="evaluate mmlu",
    )
    parser.add_argument(
        "--sigma_fuse",
        type=str,
        default="UV",
        help="sigma fuse method",
        choices=["U", "V", "UV"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=233,
        help="random seed",
    )
    parser.add_argument(
        "--r",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--first_eigen",
        action="store_true",
    )    
    parser.add_argument(
        "--save_model",
        action="store_true",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="build_adapters",
        choices=["full_decompose", "build_adapters"],
    )
    ##############################
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    # New parameters: support benchmark datasets
    parser.add_argument("--dataset_names", nargs='+', type=str, default=None, 
                       help="List of benchmark dataset names, e.g. ['MME', 'OCRBench']")
    parser.add_argument("--n_samples_per_dataset", type=int, default=16,
                       help="Number of samples per dataset")
    parser.add_argument("--data_root", type=str, 
                       default="/home/bingxing2/ailab/scx6mh7/jkl/ckpt_sum/data/vlmeval",
                       help="Root directory for benchmark data storage")
    parser.add_argument("--output_file", type=str, 
                       default="/home/bingxing2/ailab/scx6mh7/jkl/LLaVA_8_8_null_space/benchmark_seed_data",
                       help="Index of extracted benchmark data")
    parser.add_argument("--save_sampled_data", type=bool, default=True,
                       help="Whether to save the index of extracted benchmark data")

    parser.add_argument("--benchmark_cache_file", type=str,
                        default=None,
                       help="Covariance cache for extracted benchmark data")
    
    # ===== OneVision related parameters (definition) =====
    # ===== OneVision local loading parameters =====
    parser.add_argument(
        "--onevision_local_dir",
        type=str,
        default=None,
        help="Local OneVision dataset path (can be directly read by datasets.load_dataset(<path>, split='train'))"
    )
    parser.add_argument(
        "--onevision_image_dir",
        type=str,
        default=None,
        help="Directory to save exported images (will be created automatically), filename as <id>.jpg"
    )
    parser.add_argument(
        "--onevision_sources",
        type=str,
        default=None,
        help="4 source names, comma separated. e.g.: 'mathqa,iiit5k,VizWiz(MathV360K),FigureQA(MathV360K)'"
    )

    args = parser.parse_args()
    # ===== OneVision parameter normalization validation =====
    if getattr(args, "dataset_names", None) is not None and \
    any(str(x).lower() == "onevision" for x in args.dataset_names):
        # sources: comma separated → list
        if getattr(args, "onevision_sources", None) is None:
            raise ValueError("--onevision_sources must be provided, comma separated 4 sources")
        args.onevision_sources = [s.strip() for s in args.onevision_sources.split(",") if s.strip()]
        if len(args.onevision_sources) != 4:
            raise ValueError(f"--onevision_sources needs exactly 4, actual {len(args.onevision_sources)}: {args.onevision_sources}")
        # Local dataset path and image output directory are required
        if not getattr(args, "onevision_local_dir", None):
            raise ValueError("--onevision_local_dir must be provided (local dataset path)")
        if not getattr(args, "onevision_image_dir", None):
            raise ValueError("--onevision_image_dir must be provided (image save directory)")

    main(args)


