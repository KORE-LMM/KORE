
<h1 align="center"> <a href="https://arxiv.org/abs/2510.19316">KORE: Enhancing Knowledge Injection for Large Multimodal Models via Knowledge-Oriented Augmentations and Constraints</a></h1>
<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2510.19316-b31b1b.svg?logo=arXiv)](https://arxiv.org/pdf/2510.19316) [![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-KORE-blue)](https://huggingface.co/datasets/kailinjiang/KORE-74K)  [![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-KORE-blue)](https://huggingface.co/collections/kailinjiang/kore-68c54e73b6a19eece0fff381) [![code](https://img.shields.io/badge/Code-KORE-blue?logo=github)](https://github.com/KORE-LMM/KORE)  [![website](https://img.shields.io/badge/Website-KORE-orange?logo=homepage)](https://kore-lmm.github.io/) [![Slides](https://img.shields.io/badge/%F0%9F%93%8A%20Slides-KORE-BF55EC)](https://kore-lmm.github.io/KORE/slides/KORE.pdf)


</h5>






## 🛠️Requirements and Installation

```text
You can refer to https://github.com/hiyouga/LLaMA-Factory.git
```


## 💥Training

**Step 1: extract covariance matrix and reconstruct weights**
```shell
bash kore_tool/extract_covariance_matrix/step1_onevision_data.sh  -n 64 -r 235 -s 233
```
The OneVision dataset used can be downloaded from here 🤗 [LLaVA-OneVision-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data).


**Replace the dataset path in LLaMA Factory/data/dataset_info.json**

**Step 2: training**
```shell

bash kore_qwen25_vl/kore_tool/training/training_kore.sh

```

**Step 3: merge**

```shell
python kore_qwen25_vl/kore_tool/merge/merge_qwen.py --model_id training_model --save_model True --save_path merge_model
```

## 🤖Evaluation

Evaluate **EVOKE**
```shell

python kore_tool/evaluate_evoke/qwen_evoke_inference.py --input_file EVOKE/evoke_evaluation_data.jsonl --output_dir outputs/evoke_results --num_gpus 4 --model_name Qwen/Qwen2.5-VL-7B-Instruct --image_prefix /data/images/ --monitor

```



Evaluate Knowledge Retention Benchmark 
```shell
All benchmarks is based on VLMEvalKit
```

**Replace the ckpt path with the trained model here.**

https://github.com/open-compass/VLMEvalKit/blob/688e9da4a27e2691cd9a1723df6b65e5453f0889/vlmeval/config.py#L1538




