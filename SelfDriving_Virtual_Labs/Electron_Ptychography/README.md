# Packages

The following libraries are necessary for running the codes.

Firstly create a new environment and install a GPU version `py4dstem`, you can find the instructions [here](https://py4dstem.readthedocs.io/en/latest/installation.html). Below is the command while installing on `Linux` using `pip`.

```shell
conda create -n py4dstem python=3.9
conda activate py4dstem
pip install py4dstem[cuda]
```

Secondly, Please install requirements using below command.
```
pip install -r requirements.txt
```

Moreover, to use TuRBO for optimization, you need to install TuRBO first (see instruction [here](https://github.com/uber-research/TuRBO/)).

# Pipeline

**To run the self-driving virtual laboratory for electron ptychography reconstruction optimization using DANTE, see `notebooks/DANTE_VL_Electron_Ptychography_Reconstruction_Optimization.ipynb` for detailed instruction**


To have a quick start, please run the following line in terminal:

```shell
bash run.sh
```
**Note:** The environment name should be replaced by yours in `run.sh`.
