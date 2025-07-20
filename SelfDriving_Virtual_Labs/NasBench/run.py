from dataset import *
from optimizer import * 
import argparse
import random
import os

parser = argparse.ArgumentParser(description='Process inputs')
parser.add_argument('--samples', type=int, help='specify the number of samples to collect in the search')
parser.add_argument('--method', help='specify the method to search')
parser.add_argument('--random_seed', help='specify the random seed for every test')
args = parser.parse_args()
print("Max number of samples:",args.samples)
print("Optimization algorithm:",args.method)
print("Random seed:",args.random_seed)


random.seed(int(args.random_seed))
np.random.seed(int(args.random_seed))

if args.method == 'dante':
    config = {
        'exploration_weight': 0.1,
        'rollout_round': 100
    }
elif args.method == 'cma':
    config = {}
elif args.method == 'da':
    config = {}
elif args.method == 'lamcts':
    config = {}
elif args.method == 'mcmc':
    config = {
        'T_init': 0.01,
        'half_life': 200
    }
elif args.method == 'random':
    config = {}

config['samples_per_round'] = 20
config['dim'] = 14
config['random_seed'] = args.random_seed

#200 initial points
all_cells, X, y = dataset.generate_random_dataset(200)
print(X)
print(y)
os.mkdir(f'{args.method}/result/{args.random_seed}')
model = SurrogateModel(name=f'{args.method}/result/{args.random_seed}')
fx = model.fit(X, y, verbose=True)
print("")
print("="*20)
print("200 initial data points collection completed, optimization started!")
print("="*20)
print("")

# Start
result = []

optimizer = Optimizer(method=args.method, func=fx, samples=args.samples, init_cells=all_cells, init_X=X, init_y=y, config=config)

opt_result = optimizer.run()
result.extend(opt_result)
result = np.array(result)
np.save(f'{args.method}/result/result_{args.random_seed}.npy', result)

print("")
print("="*20)
print("Good job!")
print("="*20)
print("")


