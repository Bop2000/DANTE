# Colab Notebooks
We recommend you to run self-driving laboratory for cyclic peptide design using DANTE in [Colab](https://colab.research.google.com/github/Bop2000/DANTE/blob/main/notebooks/DANTE_VL_Cyclic_Peptide_Design.ipynb).
# Packages
If you want to run locally, install the following two packages first:

For install Alphafold, please refer to https://github.com/sokrypton/ColabDesign/tree/main/af

For install Rosetta, please refer to https://www.rosettacommons.org/demos/latest/tutorials/install_build/install_build

# Content
The `scripts` folder contains DANTE, MCMC and GD optimization algorithm used in the paper.

The `optimized_sequence` folder contains all the optimized sequences and their rosetta metrics.


# Pipeline

To run the self-driving virtual laboratory for cyclic peptide design, please run the following line in terminal:

```shell
unzip pdbs.zip
bash run.sh
```
**Note:** The environment name should be replaced by yours in `run.sh`. You can change the target protein and the desired length for cyclic peptide in the very beginning of `DANTE_Cyclic_Peptide_Design.py` file.
