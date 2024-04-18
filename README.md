# Derivative-free stOchastic Tree Search (DOTS)
Derivative-free tree optimization for complex systems

## Paper
Click [here](https://arxiv.org/abs/2404.04062)

## Abstract

A tremendous range of design tasks in materials, physics, and biology can be formulated as finding the optimum of an objective function depending on many parameters, without knowing its closed-form expression or the derivative. Traditional derivative-free optimization techniques often rely on strong assumptions about the objective functions, thereby often failing at optimizing non-convex systems beyond 100 dimensions. Here, we present a tree search method for derivative-free optimization that enables accelerated optimal design of high-dimensional complex systems. Specifically, Our method introduces a novel stochastic tree expansion with a dynamic upper confidence bound and short-range backpropagation to evade local optima, iteratively approximating the global optimum using machine learning models. These contributions effectively address challenging problems by achieving convergence to global optima across various benchmark functions up to 2,000 dimensions, surpassing existing methods by 10- to 20-fold. Our method demonstrates applicability to a wide range of real-world complex systems spanning materials, physics, and biology, considerably outperforming state-of-the-art algorithms. This enables efficient autonomous knowledge discovery and opens the door towards scalable self-driving virtual laboratories. Although we focus on problems within the realm of natural science, the advancements in optimization techniques achieved herein are applicable to a broader spectrum of challenges across all quantitative disciplines.

## Packages

The following libraries are necessary for running the codes.

```shell
tensorflow-gpu == 2.5.0
keras == 2.3.1
scipy == 1.10.1
numpy == 1.19.5
pandas == 1.4.4
matplotlib == 3.6.3
matplotlib-inline == 0.1.6
scikit-learn == 1.2.2
scikit-image == 0.19.3
cma == 3.3.0
tqdm == 4.59.0
seaborn == 0.12.2
openpyxl == 3.1.2
```
Please install requirements using below command.
```
pip install -r requirements.txt
```

Then install `cudnn` and `cudatoolkit`:
```
conda install conda-forge::cudatoolkit=11.2 cudnn=8.1
```

which should install in about few minutes.

## Environements
The developmental version of the package has been tested on the following systems and drivers.
- Ubuntu 18.04 and Ubuntu 22.04 
- CUDA 11.4
- cuDNN 8.1
- Driver Version 470.182.03
- RTX3090 Ti
- Python 3.9

## Pipeline

To run the DOTS on various tasks, please run the following line in terminal:

```shell
bash run.sh
```
**Note:** The codes and data of different tasks are separated in their standalone file folders, you can find a `run.sh` in each folder.

## CItation

If you find this work interesting, welcome to cite our paper!

```
[1] Wei, Y. et al. Derivative-free tree optimization for complex systems. arXiv preprint arXiv:2404.04062 (2024). https://arxiv.org/abs/2404.04062
```
