# Packages
For install Alphafold on colab, please refer to https://github.com/sokrypton/ColabDesign/tree/main/af

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
**Note:** The environment name should be replaced by yours in `run.sh`.
