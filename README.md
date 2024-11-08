# V2X-Real
[ECCV 2024] The official codebase for the paper "V2X-Real: a Largs-Scale Dataset for Vehicle-to-Everything Cooperative Perception"
[![website](https://img.shields.io/badge/Website-Explore%20Now-blueviolet?style=flat&logo=google-chrome)](https://research.seas.ucla.edu/mobility-lab/v2x-real/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2403.16034)
[![supplement](https://img.shields.io/badge/Supplementary-Material-red)](https://arxiv.org/abs/2403.16034)


This is the official implementation of ECCV 2024 paper "V2X-Real: a Largs-Scale Dataset for Vehicle-to-Everything Cooperative Perception". 
 [Hao Xiang](https://xhwind.github.io/), [Zhaoliang Zheng](https://scholar.google.com/citations?user=SyR4O7YAAAAJ&hl=en), [Xin Xia](https://scholar.google.com/citations?user=vCYqMTIAAAAJ&hl=en), [Runsheng Xu](https://derrickxunu.github.io/), [Letian Gao](https://scholar.google.com/citations?user=mz2t9m0AAAAJ&hl=en), [Zewei Zhou](https://scholar.google.com.hk/citations?user=TzhyHbYAAAAJ&hl=zh-CN), [Xu Han](https://scholar.google.com/citations?user=Ndgk55IAAAAJ&hl=en), [Xinkai Ji](https://blog.xinkaiji.cn/), [Mingxi Li](https://github.com/MingxiLii), [Zonglin Meng](), [Jin Li](), [Mingyue Lei](), [Zhaoyang Ma](), [Zihang He](), [Haoxuan Ma](), [Yunshuang Yuan](), [Yingqian Zhao](), [Jiaqi Ma](https://mobility-lab.seas.ucla.edu/)

Supported by the [UCLA Mobility Lab](https://mobility-lab.seas.ucla.edu/).


<p align="center">
<img src="images/dataset.png" width="600" alt="" class="img-responsive">
</p>


## Overview
- [Codebase Features](#codebase-features)
- [Data Download](#data-download)
- [Tutorial](#devkit-setup)
- [Citation](#citation)

## CodeBase Features
- Support both simulation and real-world cooperative perception dataset
    - [x] V2X-Real
    - [x] OPV2V
- Support multi-class multi-agent 3D object detection. 
- SOTA model supported
    - [x] [Attentive Fusion [ICRA2022]](https://arxiv.org/abs/2109.07644)
    - [x] [F-Cooper [SEC2019]](https://arxiv.org/abs/1909.06459)
    - [x] [V2VNet [ECCV2020]](https://arxiv.org/abs/2008.07519)
    - [x] [V2X-ViT [ECCV2022]](https://github.com/DerrickXuNu/v2x-vit)

## Data Download
Please check [website](https://mobility-lab.seas.ucla.edu/v2x-real/) to download the data. The data is in OPV2V format. 

After downloading the data, please put the data in the following structure:
```shell
├── v2xreal
│   ├── train
|      |── 2023-03-17-15-53-02_1_0
│   ├── validate
│   ├── test
```
## Tutorial
V2X-Real is build upon [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD). Compared to OpenCOOD, this framework supports multi-class object detection while OpenCOOD only supports single-class (i.e., vehicle) detection. V2X-Real groups object types with similar sizes to the same meta-class for conducting the learning.

#### Environment setup
Please refer to the following steps for the environment setup:
```shell
# Create conda environment (python >= 3.7)
conda create -n v2xreal python=3.7
conda activate v2xreal
# pytorch installation
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch -c conda-forge
# spconv 2.x Installation

pip install spconv-cu113
# Install other dependencies
pip install -r requirements.txt
python setup.py develop
# Install bbx nms calculation cuda version
python opencood/utils/setup.py build_ext --inplace
```

#### Running instructions
For training, please run:
```python
python opencood/tools/train_da.py --hypes_yaml hypes_yaml/xxxx.yaml --half
```
Attributes Explanations:
- `hypes_yaml`: the path for the yaml configuration of the cooperative perception models.

For inference, please run the following command: 
```python
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} --fusion_method ${FUSION_STRATEGY} [--show_vis] [--show_sequence]
```
Attributes Explanations:
- `model_dir`: the path to your saved model.
- `fusion_method`: indicate the fusion strategy, currently support 'nofusion', 'early', 'late', and 'intermediate'.
- `show_vis`: whether to visualize the detection overlay with point cloud.
- `show_sequence` : the detection results will visualized in a video stream. It can NOT be set with `show_vis` at the same time.

#### Dataset modes
To switch the dataset modes, please change the `dataset_mode` within yaml config file or the flag within the script.
Supported options:
- `vc`: V2X-Real-VC where the ego agent is fixed as the autonomous
vehicle while the collaborators include both the infrastructure and vehicle.
- `ic`: V2X-Real-IC where the infrastructure is chosen as the ego agent and the neighboring vehicles and infrastructure can collaborate with the ego infrastructure via
sharing sensing observations. The final evaluation is conducted in the ego infrastructure side.
- `v2v`: V2X-Real-V2V where only Vehicle-to-Vehicle collaboration is considered.
- `i2i`: V2X-Real-I2I where infrastructure-to-Infrastructure collaborations
are studied.

## Citation
```shell
@article{xiang2024v2x,
  title={V2X-Real: a Largs-Scale Dataset for Vehicle-to-Everything Cooperative Perception},
  author={Xiang, Hao and Zheng, Zhaoliang and Xia, Xin and Xu, Runsheng and Gao, Letian and Zhou, Zewei and Han, Xu and Ji, Xinkai and Li, Mingxi and Meng, Zonglin and others},
  journal={arXiv preprint arXiv:2403.16034},
  year={2024}
}
```
