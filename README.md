
<h1 align="center"> <a href="https://arxiv.org/abs/2510.19316">KORE: Enhancing Knowledge Injection for Large Multimodal Models via Knowledge-Oriented Augmentations and Constraints</a></h1>
<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2510.19316-b31b1b.svg?logo=arXiv)](https://arxiv.org/pdf/2510.19316) [![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-KORE-blue)](https://huggingface.co/datasets/kailinjiang/KORE-74K)  [![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-KORE-blue)](https://huggingface.co/collections/kailinjiang/kore-68c54e73b6a19eece0fff381) [![code](https://img.shields.io/badge/Code-KORE-blue?logo=github)](https://github.com/KORE-LMM/KORE)  [![website](https://img.shields.io/badge/Website-KORE-orange?logo=homepage)](https://kore-lmm.github.io/) [![Slides](https://img.shields.io/badge/%F0%9F%93%8A%20Slides-KORE-BF55EC)](https://kore-lmm.github.io/KORE/slides/KORE.pdf)




</h5>



## Table of Contents

- [Table of Contents](#table-of-contents)
- [🤗KORE](#kore)
- [🤗KORE-Augmentations](#kore-augmentations)
- [🛠️Requirements and Installation](#️requirements-and-installation)
- [💥Training](#training)
- [🤖Evaluation](#evaluation)
- [🤝 Acknowledgments](#-acknowledgments)
- [📝 Citation](#-citation)



## 🤗KORE

<div align="center">   <img src="figs\teaser.png" width="700px"> </div>

To address the challenge of balancing knowledge adaptation and retention, we propose <b>KORE</b>, a synergistic method of <u><b>K</b></u>nowledge-<u><b>oR</b></u>ient<u><b>E</b></u>d augmentations and constraints.

<div align="center">   <img src="figs\method666.jpg" width="700px"> </div>



## 🤗KORE-Augmentations

<div align="center">   <img src="figs\augmentation_comparison666.png" width="700px"> </div>

Existing methods suffer from poor generalization. General data augmentation is often "superficial and discrete" (e.g., simple rephrasing or rotation), creating isolated data points. This approach fails to build a coherent knowledge structure and offers limited support for true "knowledge internalization".

<div align="center">   <img src="figs\pipeline.png" width="700px"> </div>


<u><b>K</b></u>nowledge-<u><b>oR</b></u>ient<u><b>E</b></u>d AUGMENTATION uses an automated pipeline to convert knowledge into a "profound and structured" format. It constructs a comprehensive knowledge structure by generating "multi-rounds of dialogue" data (the trunk) and "instruction tasks" data (the branches), such as VQA and Image Caption. This process creates the KORE-74K dataset , enabling the model to achieve accurate adaptation and true "knowledge internalization" rather than just "data memorization".


You can download data 🤗 [Huggingface Dataset](https://huggingface.co/datasets/kailinjiang/KORE-74K). And the expected structure of files is:


```text
KORE-74K
|-- json/jsonl
|   |-- KORE-74K-training_data.json
|-- imgs
|   |-- imgs_of_recognition_caption_description.zip
|   |-- imgs_of_vqa
|   |   |-- split_zip_part_00
|   |   |-- split_zip_part_01
|   |   |-- split_zip_part_02
```



## 🛠️Requirements and Installation

```text
conda env create -f kore.yml
If there are any issues, you can refer to https://github.com/haotian-liu/LLaVA
```
or 

```text
conda create -n kore python=3.10 -y
cd env
pip install -r kore.txt
```

## 💥Training

**Step 1: extract covariance matrix and reconstruct weights**
```shell
bash kore_tool/extract_covariance_matrix/step1_benchmark.sh -d "MME MMBench_DEV_EN" -n 128 -r 235 -s 233

The selection of -d refers to 'DATASET_CONFIG' in benchmark_load.py, like: MME, HallusionBench, MathVision......

bash kore_tool/extract_covariance_matrix/step1_onevision_data.sh -d "onevision" -n 64 -r 235 -s 233
```
The OneVision dataset used can be downloaded from here 🤗 [LLaVA-OneVision-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data).




**Step 2: training**
```shell
bash kore_tool/training/training_kore.sh --data_path KORE-74K-training_data.json --output_dir train_ckpt/kore_epoch1 --num_train_epochs 1 --swanlab_project "kore" --swanlab_experiment_name "epoch1"

--lora_null_v1 True does not freeze the 'A' matrix, whereas --lora_null_v2 True does.
```


**Step 3: merge**

```shell
python kore_tool/merge/merge_llava.py --model_id training_model --save_model True --save_path merge_model
```



## 🤖Evaluation

Evaluate **EVOKE**
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash kore_tool/evaluate_evoke/evoke.sh -c /path/to/checkpoint -o /path/to/output -q EVOKE/evoke_evaluation_data.jsonl
```

Evaluate Knowledge Retention Benchmark (**MME,MMBench,POPE,ScienceQA** is based on the Llava framework itself)
```shell
bash kore_tool/evaluate_retention_benchmark/mmbench.sh -m /path/to/model/checkpoint
bash kore_tool/evaluate_retention_benchmark/mme.sh -m /path/to/model/checkpoint
bash kore_tool/evaluate_retention_benchmark/pope.sh -m /path/to/model/checkpoint
bash kore_tool/evaluate_retention_benchmark/sqa.sh -m /path/to/model/checkpoint
```

Evaluate Knowledge Retention Benchmark 
```shell
Other benchmarks is based on VLMEvalKit
```

Replace the ckpt path with the trained model here.
https://github.com/open-compass/VLMEvalKit/blob/688e9da4a27e2691cd9a1723df6b65e5453f0889/vlmeval/config.py#L709


## 🤝 Acknowledgments
We thank the following open-source projects for making this work possible:
- [LLaVA](https://github.com/haotian-liu/LLaVA.git) for the model training framework.
- [CorDA](https://github.com/iboing/CorDA) and [LoRA-Null](https://github.com/HungerPWAY/LoRA-Null.git) for the constraint fine-tuning framework.
- [EVOKE](https://github.com/EVOKE-LMM/EVOKE) for the knowledge adaptation evaluation.
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit.git) for the knowledge retention evaluation.
- [MCITlib](https://github.com/Ghy0501/MCITlib) and [CoIN](https://github.com/zackschen/CoIN.git) for the continual learning methods framework.



## 📝 Citation
If you find our paper and code useful in your research, please consider giving a star ⭐ and citation 📝 :)

```bibtex
@article{jiang2025kore,
  title = {KORE: Enhancing Knowledge Injection for Large Multimodal Models via Knowledge-Oriented Augmentations and Constraints},
  author={Jiang, Kailin and Jiang, Hongbo and Jiang, Ning and Gao, Zhi and Bi, Jinhe and Ren, Yuchen and Li, Bin and Du, Yuntao and Liu, Lei and Li, Qing},
  journal={arXiv preprint arXiv:2510.19316},
  year={2025}
  url = {https://arxiv.org/abs/2510.19316}
}
```



