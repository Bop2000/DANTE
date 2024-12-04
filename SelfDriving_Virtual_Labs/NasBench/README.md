# Pipeline

**The folder contains everything needed to run the self-driving virtual laboratory for neural architecture search using DANTE.**

To have a quick start, please run the following line in terminal:

```shell
bash run.sh
```

**Note:** The environment name should be replaced by yours in `run.sh`. You can change the number of samples in optimization process, or the optimization method by changing the parameters in `python3 run.py --samples 200 --method random`. Currently, it compiles `dante/cma/da/lamcts/mcmc/random` algorithms.