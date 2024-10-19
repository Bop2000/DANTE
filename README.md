# Deep Active learning with Neural-surrogate-guided Tree Exploration (DANTE)

## What is DANTE
Click [here](https://arxiv.org/abs/2404.04062)

## Introduction

Inferring optimal solutions from limited data is considered the ultimate goal in scientific discovery. Artificial intelligence offers a promising avenue to greatly accelerate this process. Existing methods often depend on large datasets, strong assumptions about objective functions, and classic machine learning techniques, restricting their effectiveness to low-dimensional problems.

We present a deep active learning pipeline that combines deep neural network and a novel tree search to find superior solutions in high-dimensional complex problems characterized by limited data availability. Our pipeline iteratively approaches the optimum using a neural surrogate and introduces new search mechanisms to bypass the local optimum. These contributions enable our pipeline to achieve superior solutions across diverse problems with up to 2,000 dimensions, whereas existing methods are limited to 100 dimensions and require 10 times more data points. Our pipeline demonstrates wide applicability, discovering superior solutions in various domain science problems. This advancement enables data-efficient knowledge discovery and opens the path towards scalable self-driving laboratories. Although we focus on problems within the realm of scientific domain, the advancements achieved herein are applicable to a broader spectrum of challenges across all quantitative disciplines. 
![alt text](https://github.com/Bop2000/DOTS/blob/main/assets/flowchart.jpg)

## Installation

The code requires `python>=3.9`. Installation Tensorflow and Keras with CUDA support is stroongly recommended.

Install MLTS:

```
pip install git+https://github.com/Bop2000/MLTS.git
```

or clone the repository to local devices:

```
git clone git@github.com:Bop2000/MLTS.git
cd DOTS; pip install -e .
```

## Environments
The developmental version of the package has been tested on the following systems and drivers.
- Ubuntu 18.04 and Ubuntu 22.04 
- CUDA 11.4
- cuDNN 8.1
- Driver Version 470.182.03
- RTX3090 Ti

## Pipeline

To run the DOTS on various tasks, please run the following line in terminal:

```shell
bash run.sh
```
**Note:** The codes and data of different tasks are separated in their standalone file folders, you can find a `run.sh` in each folder.

To run the self-driving virtual labs, find `run.sh` in `Architected Materials`, `CCAs`, `Cyclic peptide design`, and `Electron Ptychography` folders.

To conduct evaluations on synthetic functions, find `run.sh` in `Evaluations on Synthetic Functions/Exact function` and `Evaluations on Synthetic Functions/Surrogate model` folders.

## Citation

If you find this work interesting, welcome to cite our paper!

```
@article{wei2024derivative,
  title={Derivative-free tree optimization for complex systems},
  author={Wei, Ye and Peng, Bo and Xie, Ruiwen and Chen, Yangtao and Qin, Yu and Wen, Peng and Bauer, Stefan and Tung, Po-Yen},
  journal={arXiv preprint arXiv:2404.04062},
  year={2024}
}
```
