# Pipeline

**To conduct evaluations on synthetic functions using surrogate model with DANTE, see `notebooks/DANTE_Rastrigin_200d.ipynb` for detailed instruction**

**Note:** If you can not open it in Github, please download it.

####################################################################################

To have a quick start, please run the following line in terminal:

```shell
bash run.sh
```
**Note:** The environment name should be replaced by yours in `run.sh`. 

--func (specify the test function) can be: ackley rastrigin rosenbrock schwefel michalewicz griewank levy

--method (specify the method to search) can be: DANTE here, but extended to DANTE-Greedy DANTE-eGreedy Random DualAnnealing DifferentialEvolution CMA-ES in `more_algorithms_available`

--dims (specify the problem dimensions) can be: any integer more than 1

--samples (specify the number of samples to collect in the search) can be: any integer more than 0



# Evaluations on synthetic functions using surrogate model

We run extensive tests on well-known non-convex functions (Ackley, Rastrigin, Rosenbrock) of diverse types using surrogate model predictions and compare the performance of DANTE with other state-of-the-art algorithms. DANTE-based methods outperform other benchmark methods.

![alt text](https://github.com/Bop2000/DANTE/blob/main/assets/figure_3.png)
