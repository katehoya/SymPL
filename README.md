# Keep it SymPL: Symbolic Projective Layout for Allocentric Spatial Reasoning in Vision-Language Models

<p align="center">
  <a href="https://sites.google.com/khu.ac.kr/jjy-fine/home" target="_blank"><strong>Jaeyun Jang</strong></a> ·
  <a href="https://seunghui-shin.github.io/" target="_blank"><strong>Seunghui Shin</strong></a> ·
  <a href="https://airlab.khu.ac.kr/" target="_blank"><strong>Taeho Park</strong> ·
  <a href="https://sites.google.com/view/hyoseok-hwang" target="_blank"><strong>Hyoseok Hwang</strong></a><sup>*</sup>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.19117">
    <img src="https://img.shields.io/badge/arXiv-2602.19117-b31b1b.svg" />
  </a>
  <a href="https://airlabkhu.github.io/SymPL/">
    <img src="https://img.shields.io/badge/Project-SymPL-yellow" />
  </a>
</p>

## Abstract
<p align="justify">
Perspective-aware spatial reasoning involves understanding spatial relationships from specific viewpoints—either egocentric (observer-centered) or allocentric (object-centered). While vision–language models (VLMs) perform well in egocentric settings, their performance deteriorates when reasoning from allocentric viewpoints, where spatial relations must be  nferred from the perspective of objects within the scene. In this study, we address this underexplored challenge by introducing Symbolic Projective Layout (SymPL), a framework that reformulates allocentric reasoning into symbolic-layout forms that VLMs inherently handle well. By leveraging four key factors—projection, abstraction, bipartition, and localization—SymPL converts allocentric questions into structured symbolic-layout representations. Extensive experiments demonstrate that this reformulation substantially improves performance in both allocentric and egocentric tasks, enhances robustness under visual illusions and multi-view scenarios, and that each component contributes critically to these gains. These results show that SymPL provides an effective and principled approach for addressing complex perspective-aware spatial reasoning.
</p>


![](docs/static/images/introduction.png)

## Figures 

## Main Results

## Installation

## Usage

## Inference
To run inference on an RGB and create 3D strands use:

    $ ./inference_difflocks.py \
		--img_path=./samples/medium_11.png \
		--out_path=./outputs_inference/ 

You also have options to export a `.blend` file and an alembic file by specifying `--blender_path` and `--export_alembic` in the above script. 
Note that the blender path corresponds to the blender executable with version 4.1.1. It will likely not work with other versions. 

	
## Train StrandVAE 
To train the strandVAE model: 

	$ ./train_strandsVAE.py --dataset_path=<DATASET_PATH> --exp_info=<EXP_NAME>

it will start training and outputting tensorboard logs in `./tensorboard_logs`


## Train DiffLocks diffusion model 
To train the diffusion model: 

	$ ./train_scalp_diffusion.py \
		--config ./configs/config_scalp_texture_conditional.json \
		--batch-size 4 \
		--grad-accum-steps 4 \
		--mixed-precision bf16 \
		--use-tensorboard \
		--save-checkpoints \
		--save-every 100000 \
		--compile \
		--dataset_path=<DATASET_PATH> \
		--dataset_processed_path=<DATASET_PATH_PROCESSED>
		--name <EXP_NAME> 

it will start training and outputting tensorboard logs in `./tensorboard_logs`. 
Start training on multiple GPUs by first running:

	$ accelerate config

followed by pre-pending `accelerate launch` to the previous training script:

	$ accelerate launch ./train_scalp_diffusion.py \
		--config ./configs/config_scalp_texture_conditional.json \
		--batch-size 4 \
		<ALL_THE_OTHER_OPTIONS_AS_SPECIFIED_ABOVE>

You would probably to adjust the `batch-size` and `grad-accum-step` depending on the number of GPUs you have. 
