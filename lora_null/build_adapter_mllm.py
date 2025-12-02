import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM
from adapterlib.datautils import get_calib_data, get_calib_data_mllm
from adapterlib.act_aware_utils import calib_input_distribution, calib_fisher_info, calib_cov_distribution
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

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    # )

    # collect data
    # calib_loader = get_calib_data(args.calib_dataset, tokenizer, model_id, args.calib_loader_size, seed=args.seed) #256, 128
    # get_calib_data_mllm 返回 (model, processor, data_loader)
    model, _, calib_loader = get_calib_data_mllm(args)
    
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
        #原本的操作，需要修改bug，明天记得改好！！！！！！！！！！！！！！！
        #原本的操作，需要修改bug，明天记得改好！！！！！！！！！！！！！！
        #原本的操作，需要修改bug，明天记得改好！！！！！！！！！！！！！！
        calib_cov_distribution(
            model, calib_loader, args.use_cache, args.calib_dataset, args.calib_loader_size, args.dataset_names, args.n_samples_per_dataset, args.benchmark_cache_file, seed=args.seed
        )

        # calib_infoseek_cov_distribution(
        #     model, calib_loader, args.use_cache, args.calib_dataset, args.calib_loader_size, args.dataset_names, args.n_samples_per_dataset, args.benchmark_cache_file, seed=args.seed
        # )

        #calib_singular_vectors(
        #    model, calib_loader, args.use_cache, args.calib_dataset, args.calib_loader_size, seed=args.seed
        #)
    elif args.singular_aware_2:
        print('Collecting covariance data for Singular_aware ...')
        calib_cov_distribution(
            model, calib_loader, args.use_cache, args.calib_dataset, args.calib_loader_size, seed=args.seed
        )
    else:
        print('Use the normal SVD ...')

    #  perform decomposition
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
    parser.add_argument(
        "--proj_aware",
        action="store_true",
    )
    parser.add_argument(
        "--proj_x_aware",
        action="store_true",
    )
    parser.add_argument(
        "--ori_proj_x_aware",
        action="store_true",
    )

    parser.add_argument(
        "--cov_aware_wo_inv",
        action="store_true",
    )
    parser.add_argument(
        "--proj_x_t_aware",
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

    # 新增参数：支持benchmark数据集
    parser.add_argument("--dataset_names", nargs='+', type=str, default=None, 
                       help="benchmark数据集名称列表，如 ['MME', 'OCRBench']")
    parser.add_argument("--n_samples_per_dataset", type=int, default=16,
                       help="每个数据集采样的样本数")
    parser.add_argument("--data_root", type=str, 
                       default="/hkfs/work/workspace/scratch/lmu_chd4938-MINED_26/MINED_26/data_ckpt/dataset/vlmeval",
                       help="benchmark数据存储根目录")
    parser.add_argument("--output_file", type=str, 
                       default="/home/bingxing2/ailab/scx6mh7/jkl/LLaVA_8_8_null_space/benchmark_seed_data",
                       help="benchmark数据取出数据的index")
    parser.add_argument("--save_sampled_data", type=bool, default=True,
                       help="是否要保存benchmark数据取出数据的index")

    parser.add_argument("--benchmark_cache_file", type=str,
                        default=None,
                       help="benchmark数据取出数据的协方差cache")
    
    # ===== OneVision 相关参数（定义） =====
    # ===== OneVision 本地加载参数 =====
    parser.add_argument(
        "--onevision_local_dir",
        type=str,
        default=None,
        help="本地 OneVision 数据集路径（可被 datasets.load_dataset(<path>, split='train') 直接读取）"
    )
    parser.add_argument(
        "--onevision_image_dir",
        type=str,
        default=None,
        help="保存导出图片的目录（会自动创建），文件名为 <id>.jpg"
    )
    parser.add_argument(
        "--onevision_sources",
        type=str,
        default=None,
        help="4 个 source 名，逗号分隔。例如：'mathqa,iiit5k,VizWiz(MathV360K),FigureQA(MathV360K)'"
    )



    args = parser.parse_args()

    # ===== OneVision 参数归一化校验 =====
    if getattr(args, "dataset_names", None) is not None and \
    any(str(x).lower() == "onevision" for x in args.dataset_names):
        # sources: 逗号分隔 → list
        if getattr(args, "onevision_sources", None) is None:
            raise ValueError("--onevision_sources 必须提供，逗号分隔 4 个 source")
        args.onevision_sources = [s.strip() for s in args.onevision_sources.split(",") if s.strip()]
        if len(args.onevision_sources) != 4:
            raise ValueError(f"--onevision_sources 需要恰好 4 个，实际 {len(args.onevision_sources)}: {args.onevision_sources}")

        # 本地数据集路径与图片输出目录必填
        if not getattr(args, "onevision_local_dir", None):
            raise ValueError("--onevision_local_dir 必须提供（本地数据集路径）")
        if not getattr(args, "onevision_image_dir", None):
            raise ValueError("--onevision_image_dir 必须提供（图片保存目录）")

    main(args)


    # args = parser.parse_args()

    # main(args)
