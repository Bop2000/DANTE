#--func can be: ackley rastrigin rosenbrock schwefel michalewicz griewank levy
#--method can be: DOTS  DOTS-Greedy DOTS-eGreedy Random DualAnnealing DifferentialEvolution CMA-ES

source $HOME/.bashrc

conda activate tf_25 #your environment

python3 run.py --func rosenbrock --dims 100 --iterations 100000 --method DOTS

