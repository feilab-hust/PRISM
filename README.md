# PRISM: Physics-informed Regularized Inversion of Spatial Mixing in fluorescence microscopy

This repository provides the PyTorch implementation of **PRISM** (**P**hysics-informed **R**egularized **I**nversion of **S**patial **M**ixing), a self-supervised restoration framework for fluorescence microscopy images degraded by coupled background contamination and diffraction-limited optical blur.

PRISM models an observed fluorescence image as a spatially weighted mixture of a restored fluorescence signal and a smoothly varying background component governed by a spatial transmittance map. A dual-branch neural network jointly estimates the restored image and the transmittance map under a physics-informed forward model. Frequency-band consistency and dark-channel sparsity are used to regularize the self-supervised inversion without paired low- and high-quality training data.

> Paper title: **Physics-informed Regularized Inversion of Spatial Mixing (PRISM) in fluorescence microscopy**

---

## Contents

- [Main features](#main-features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Repository structure](#repository-structure)
- [Data and PSF format](#data-and-psf-format)
- [Usage](#usage)
  - [2D training](#2d-training)
  - [2D inference](#2d-inference)
  - [3D training](#3d-training)
  - [3D inference](#3d-inference)
- [Output files](#output-files)
- [Citation](#citation)
- [Contact](#contact)

---

## Main features

- Self-supervised restoration without paired ground-truth images.
- Physics-informed spatial mixing model for coupled background contamination and optical blur.
- Dual-branch estimation of restored fluorescence signal and spatial transmittance map.
- Frequency-band observation consistency and dark-channel sparsity regularization.
- Separate implementations for 2D images and 3D volumetric stacks.
- Patch-based training and block-wise full-field inference for large microscopy data.

---

## Requirements

The code was developed with PyTorch and CUDA-enabled GPUs. The 3D version is memory intensive and is recommended to run on a modern NVIDIA GPU.

Recommended environment:

- Python 3.12
- PyTorch 2.5.1
- CUDA 12.4 compatible PyTorch runtime
- NumPy, SciPy, tifffile, matplotlib, tqdm
- timm, einops, Kornia

A Conda environment file is provided as `PRISM_environment.yaml`.

---

## Installation

Clone the repository and create the Conda environment:

```bash
git clone https://github.com/<your-lab>/PRISM.git
cd PRISM
conda env create -f PRISM_environment.yaml
conda activate prism
```

If your local CUDA driver does not support CUDA 12.4, install a PyTorch build compatible with your system from the official PyTorch website and then install the remaining Python packages listed in `PRISM_environment.yaml`.

---

## Repository structure

The expected repository layout is:

```text
PRISM/
├── 2D/
│   ├── train.py                 # 2D self-supervised training
│   ├── inference.py             # 2D block-wise inference
│   ├── model.py                 # network definitions
│   ├── loss.py                  # loss functions
│   ├── data_utils.py            # patch extraction and data preprocessing
│   ├── transmittance.py         # transmittance estimation modules
│   └── utils.py                 # utility functions
├── 3D/
│   ├── train.py                 # 3D self-supervised training
│   ├── inference.py             # 3D block-wise inference
│   ├── model.py
│   ├── loss.py
│   ├── data_utils.py
│   ├── transmittance.py
│   └── utils.py
├── data/
│   ├── 2D/example/              # example 2D .tif images
│   ├── 3D/example/              # example 3D .tif stacks
│   ├── psf/                     # example PSF
├── weights/                     # pretrained or trained model weights (.pth)
├── results/                     # inference outputs
├── PRISM_environment.yaml
└── README.md
```

---

## Data and PSF format

### 2D data

- Input: a folder containing `.tif` or `.tiff` images.
- Expected shape for each image: `[H, W]`.

### 3D data

- Input: a folder containing volumetric `.tif` or `.tiff` stacks.
- Expected shape for each stack: `[Z, H, W]`.

### PSF
- PSF: a `.tif` kernel. The training script normalizes the PSF to unit sum before use.

The current preprocessing normalizes each image or patch by its maximum intensity. Use non-negative fluorescence images whenever possible.

---

## Usage

The current scripts contain default hyperparameters for the reported 2D and 3D experiments. Before running, update the input directory, PSF path, checkpoint path and output directory in the corresponding script, or apply the command-line path refactor described in [Preparing the code for GitHub release](#preparing-the-code-for-github-release).

### 2D training

Edit the path variables in `2D/train.py`:

```python
x_blur_dir = "../data/2D/example/input"
kernel_path = "../data/2D/example/psf.tif"
model_path = "../weights/2D_example"
best_weight_path = ""  # optional; set to a previous checkpoint for fine-tuning
```

Then run:

```bash
cd 2D
python train.py
```

or:
```bash
python 2D/train.py --input_dir data/2D/example/input --psf_path data/2D/example/psf.tif --output_dir weights/2D_example --gpu 0
```

Important default parameters in `2D/train.py` include:

- `scale = 2`: lateral upsampling factor.
- `crop_size = 128`: training patch size.
- `num_per_img = 20`: number of random patches sampled from each image before high-variance selection.
- `top_percent = 0.1`: fraction of high-variance patches retained.
- `back_flag = True`: enables joint background/transmittance estimation.
- `num_epochs = 100` when `back_flag = True`.

For a two-stage workflow, first train with `back_flag = False` to initialize the restoration branch, then train with `back_flag = True` using the first-stage checkpoint as `best_weight_path`.

### 2D inference

Edit the path variables in `2D/Inference.py`:

```python
weight_path = "../weights/2D_example/best_model2.pth"
path_x = "../data/2D/example/input"
save_path = "../results/2D_example"
```

Then run:

```bash
cd 2D
python Inference.py
```
or:
```bash
python 2D/Inference.py --input_dir data/2D/example/input --weight_path weights/2D_example/best_model2.pth --output_dir results/2D_example --gpu 0
```

The script performs block-wise inference with overlapping windows and saves restored images as 16-bit TIFF files.

### 3D training

Edit the path variables in `3D/train.py`:

```python
x_blur_dir = "../data/3D/example/input"
kernel_path = "../data/3D/example/psf.tif"
model_path = "../weights/3D_example"
best_weight_path = ""  # optional; set to a previous checkpoint for fine-tuning
```

Then run:

```bash
cd 3D
python train.py
```
or:
```bash
python 3D/train.py --input_dir data/3D/example/input --psf_path data/3D/example/psf.tif --output_dir weights/3D_example --gpu 0
```

Important default parameters in `3D/train.py` include:

- `scale = 2`: lateral upsampling factor.
- `scale_axial = 1`: axial upsampling factor during training by default.
- `crop_size = 96`: lateral patch size.
- `crop_depth = 16`: axial patch depth.
- `num_per_img = 800`: number of random patches sampled from each stack before high-variance selection.
- `top_percent = 0.1`: fraction of high-variance patches retained.
- `back_flag = True`: enables joint background/transmittance estimation.

### 3D inference

Edit the path variables in `3D/Inference.py`:

```python
weight_path = "../weights/3D_example/best_model2.pth"
path_x = "../data/3D/example/input"
save_path = "../results/3D_example"
```

Then run:

```bash
cd 3D
python Inference.py
```
or:
```bash
python 3D/Inference.py --input_dir data/3D/example/input --weight_path weights/3D_example/best_model2.pth --output_dir results/3D_example --gpu 0
```

The 3D inference script performs block-wise restoration with overlapping 3D windows. The default block settings are:

- `block_size = 192`
- `block_depth = 24`
- `overlap = 16`
- `overlap_depth = 8`

Adjust these values according to GPU memory and stack size.

---

## Output files

During training, the scripts generate:

- `cached_data.pt`: cached training patches.
- `cached_data/`: optional saved training patches.
- `cached_data/final/`: restored patch outputs from the trained model.
- `cached_data/final/back/`: estimated background outputs when `back_flag = True`.
- `best_model2.pth`: checkpoint with the best training loss.
- `restoration_model2.pth`: final trained model.
- `loss_curve2.png`: training loss curve.

During inference, restored TIFF images are saved to the specified `save_path`.

---

## Citation

If you use PRISM in your research, please cite the corresponding manuscript:

```text
Physics-informed Regularized Inversion of Spatial Mixing (PRISM) in fluorescence microscopy. 
```

---


## Contact

For questions, please open an issue in this repository or contact:

```text
2675260749@qq.com
```
