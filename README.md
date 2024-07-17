# Derivative-free stOchastic Tree Search (DOTS)

## Paper
Click [here](https://arxiv.org/abs/2404.04062)

## Introduction

A tremendous range of design tasks in materials, physics, and biology can be formulated as finding the optimum of an objective function depending on many parameters, without knowing its closed-form expression or the derivative. Traditional derivative-free optimization techniques often rely on strong assumptions about the objective functions, thereby often failing at optimizing non-convex systems beyond 100 dimensions. Here, we present a tree search method for derivative-free optimization that enables accelerated optimal design of high-dimensional complex systems. Specifically, Our method introduces a novel stochastic tree expansion with a dynamic upper confidence bound and short-range backpropagation to evade local optima, iteratively approximating the global optimum using machine learning models. These contributions effectively address challenging problems by achieving convergence to global optima across various benchmark functions up to 2,000 dimensions, surpassing existing methods by 10- to 20-fold. Our method demonstrates applicability to a wide range of real-world complex systems spanning materials, physics, and biology, considerably outperforming state-of-the-art algorithms. This enables efficient autonomous knowledge discovery and opens the door towards scalable self-driving virtual laboratories. Although we focus on problems within the realm of natural science, the advancements in optimization techniques achieved herein are applicable to a broader spectrum of challenges across all quantitative disciplines.
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
