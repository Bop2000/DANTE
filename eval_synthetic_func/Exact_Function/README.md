# Pipeline

**To conduct evaluations on synthetic functions using exact functions with DOTS, please run the following line in terminal**


```shell
bash run.sh
```
**Note:** The environment name should be replaced by yours in `run.sh`. 

--func (specify the test function) can be: ackley rastrigin rosenbrock schwefel michalewicz griewank levy

--method (specify the method to search) can be: DOTS DOTS-Greedy DOTS-eGreedy Random DualAnnealing DifferentialEvolution CMA-ES

--dims (specify the problem dimensions) can be: any integer more than 1

--iterations (specify the number of samples to collect in the search) can be: any integer more than 0
