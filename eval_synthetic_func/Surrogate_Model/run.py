from functions import *
import argparse
import os
import numpy as np
from dots.search_method import *
from dots.NN import *
parser = argparse.ArgumentParser(description='Process inputs')
parser.add_argument('--func', help='specify the test function')
parser.add_argument('--dims', type=int, help='specify the problem dimensions')
parser.add_argument('--samples', type=int, help='specify the number of samples to collect in the search')
parser.add_argument('--method', help='specify the method to search')
args = parser.parse_args()

f = None
if args.func == 'ackley':
    assert args.dims > 0
    f = Ackley(dims =args.dims)
    fx = Surrogate(dims =args.dims, name=args.method+'-Ackley', f=f, iters = args.samples)
elif args.func == 'rastrigin':
    assert args.dims > 0
    f = Rastrigin(dims =args.dims)
    fx = Surrogate(dims =args.dims, name=args.method+'-Rastrigin', f=f, iters = args.samples)
elif args.func == 'rosenbrock':
    assert args.dims > 0
    f = Rosenbrock(dims =args.dims)
    fx = Surrogate(dims =args.dims, name=args.method+'-Rosenbrock', f=f, iters = args.samples)
elif args.func == 'levy':
    f = Levy(dims = args.dims)
    fx = Surrogate(dims =args.dims, name=args.method+'-Levy', f=f, iters = args.samples)
elif args.func == 'schwefel':
    f = Schwefel(dims = args.dims)
    fx = Surrogate(dims =args.dims, name=args.method+'-Schwefel', f=f, iters = args.samples)
elif args.func == 'michalewicz':
    f = Michalewicz(dims = args.dims)
    fx = Surrogate(dims =args.dims, name=args.method+'-Michalewicz', f=f, iters = args.samples)
elif args.func == 'griewank':
    f = Griewank(dims = args.dims)
    fx = Surrogate(dims =args.dims, name=args.method+'-Griewank', f=f, iters = args.samples)

else:
    print('function not defined')
    os._exit(1)

nn = model_training(f=args.func, dims=args.dims)

assert args.dims > 0
assert f is not None
assert args.samples > 0

lower = f.lb
upper = f.ub

bounds = []
for idx in range(0, len(f.lb) ):
    bounds.append( ( float(f.lb[idx]), float(f.ub[idx])) )


#200 initial points
aaa = np.arange(f.lb[0], f.ub[0] + f.turn, f.turn).round(5)
input_X = np.random.choice(aaa,size=(200, args.dims))
input_y = []
input_y2 = []
for i in input_X:
    y1, y2 = fx(i)
    input_y.append(y1)
    input_y2.append(y2)
input_X = np.array(input_X)
input_y2 = np.array(input_y2)
print("")
print("="*20)
print("200 initial data points collection completed, optimization started!")
print("="*20)
print("")

if args.func == 'ackley' or args.func == 'rastrigin':
    rollout_round = 200
else:
    rollout_round = 100

if args.method == 'DOTS':
    if args.func == 'ackley':
        ratio = 0.1
    elif args.func == 'rastrigin':
        ratio = 1
    elif args.func == 'rosenbrock':
        ratio = 1
    else:
        ratio = 1
    for i in range(args.samples//20):
        model = nn(input_X,input_y2)
        optimizer = MCTS_ubt(f=f, model=model, name = args.func)
        top_X = optimizer.rollout(input_X,input_y2,rollout_round,ratio,i)
        top_y = []
        for xx in top_X:
            y1, y2 = fx(xx)
            top_y.append(y2)
        top_y = np.array(top_y)
        input_X=np.concatenate((input_X,top_X),axis=0)
        input_y2=np.concatenate((input_y2,top_y))
        
        if f(input_X[np.argmax(input_y2)])[0] == 0:
            break

elif args.method == 'DOTS-Greedy':
    for i in range(args.samples//20):
        model = nn(input_X,input_y2)
        optimizer = MCTS_Greedy(f = f, dims = args.dims, model=model, name = args.func)
        top_X = optimizer.rollout(input_X,input_y2, rollout_round)
        top_y = []
        for xx in top_X:
            y1, y2 = fx(xx)
            top_y.append(y2)
        top_y = np.array(top_y)
        input_X=np.concatenate((input_X,top_X),axis=0)
        input_y2=np.concatenate((input_y2,top_y))
        
        if f(input_X[np.argmax(input_y2)]) == 0:
            break

elif args.method == 'DOTS-eGreedy':
    for i in range(args.samples//20):
        model = nn(input_X,input_y2)
        optimizer = MCTS_eGreedy(f = f, dims = args.dims, model=model, name = args.func)
        top_X = optimizer.rollout(input_X,input_y2, rollout_round)
        top_y = []
        for xx in top_X:
            y1, y2 = fx(xx)
            top_y.append(y2)
        top_y = np.array(top_y)
        input_X=np.concatenate((input_X,top_X),axis=0)
        input_y2=np.concatenate((input_y2,top_y))
        
        if f(input_X[np.argmax(input_y2)]) == 0:
            break

elif args.method == 'Random':
    aaa = np.arange(f.lb[0], f.ub[0] + f.turn, f.turn)
    init_X = np.random.choice(aaa,size=(args.samples,args.dims))
    for i in init_X:
        init_y = fx(i)


elif args.method == 'DualAnnealing':
    for i in range(args.samples//20):
        model = nn(input_X,input_y2)
        optimizer = DualAnnealing(f = f, dims = args.dims, model=model, name = args.func)
        optimizer.mode = 'fast' # 'fast' or 'origin'
        print('This optimization is based on a ',optimizer.mode, ' mode Dual Annealing optimizer')
        top_X = optimizer.rollout(input_X,input_y2, rollout_round)
        top_y = []
        for xx in top_X:
            y1, y2 = fx(xx)
            top_y.append(y2)
        top_y = np.array(top_y)
        input_X=np.concatenate((input_X,top_X),axis=0)
        input_y2=np.concatenate((input_y2,top_y))
        
        if f(input_X[np.argmax(input_y2)]) == 0:
            break

elif args.method == 'DifferentialEvolution':
    for i in range(args.samples//20):
        model = nn(input_X,input_y2)
        optimizer = DifferentialEvolution(f = f, dims = args.dims, model=model, name = args.func)
        optimizer.mode = 'fast' # 'fast' or 'origin'
        print('This optimization is based on a ',optimizer.mode, ' mode Differential Evolution optimizer')
        top_X = optimizer.rollout(input_X,input_y2, rollout_round)
        top_y = []
        for xx in top_X:
            y1, y2 = fx(xx)
            top_y.append(y2)
        top_y = np.array(top_y)
        input_X=np.concatenate((input_X,top_X),axis=0)
        input_y2=np.concatenate((input_y2,top_y))
        
        if f(input_X[np.argmax(input_y2)]) == 0:
            break

elif args.method == 'CMA-ES':
    for i in range(args.samples//20):
        model = nn(input_X,input_y2)
        optimizer = CMAES(f = f, dims = args.dims, model=model, name = args.func)
        optimizer.mode = 'fast' # 'fast' or 'origin'
        print('This optimization is based on a ',optimizer.mode, ' mode CMA-ES optimizer')
        top_X = optimizer.rollout(input_X,input_y2, rollout_round)
        top_y = []
        for xx in top_X:
            y1, y2 = fx(xx)
            top_y.append(y2)
        top_y = np.array(top_y)
        input_X=np.concatenate((input_X,top_X),axis=0)
        input_y2=np.concatenate((input_y2,top_y))
        
        if f(input_X[np.argmax(input_y2)]) == 0:
            break

