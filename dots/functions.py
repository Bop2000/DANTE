import numpy as np
import json
import os


class tracker:
    def __init__(self, foldername):
        self.counter   = 0
        self.results   = []
        self.curt_best = float("inf")
        self.curt_best_x = None
        self.foldername = foldername
        try:
            os.mkdir(foldername)
        except OSError:
            print ("Creation of the directory %s failed" % foldername)
        else:
            print ("Successfully created the directory %s " % foldername)
        
    def dump_trace(self):
        # trace_path = self.foldername + '/result' + str(len( self.results) )
        trace_path = self.foldername + '/result'
        final_results_str = json.dumps(self.results)
        with open(trace_path, "a") as f:
            f.write(final_results_str + '\n')
            
    def track(self, result, x = None, iters = 1000):
        if result < self.curt_best:
            self.curt_best = result
            self.curt_best_x = x
        print("")
        print("="*10)
        print("iteration:", self.counter, "total samples:", len(self.results) )
        print("="*10)
        print("current best f(x):", self.curt_best)
        print("current best x:", np.around(self.curt_best_x, decimals=4))
        self.results.append(self.curt_best)
        self.counter += 1
        if len(self.results) == iters:
            self.dump_trace()
        elif round(self.curt_best,5) == 0:
            self.dump_trace()
        
class Levy:
    def __init__(self, dims=1, turn = 0.1):
        self.dims    = dims
        self.lb      = -10 * np.ones(dims)
        self.ub      =  10 * np.ones(dims)
        self.counter = 0
        self.turn    = turn

    def __call__(self, x):
        x = np.array(x / self.turn).round(0) * self.turn
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        # assert np.all(x <= self.ub) and np.all(x >= self.lb)
        d = len(x) 
        w = 1 + (x - 1) / 4 
        term1 = (np.sin(np.pi * w[0]))**2
        term3 = ((w[d-1] - 1)**2) * (1 + (np.sin(2 * np.pi * w[d-1]))**2)
        sum_term = np.sum(((w[:d-1] - 1)**2) * (1 + 10 * (np.sin(np.pi * w[:d-1] + 1))**2))
        
        return term1 + sum_term + term3
    
    def multi(self,X):
        X = np.array(X / self.turn).round(0) * self.turn
        w = 1 + (X - 1) / 4
        term1 = np.sin(np.pi * w[:, 0])**2
        term3 = (w[:, -1] - 1)**2 * (1 + np.sin(2 * np.pi * w[:, -1])**2)
        sum_term = np.sum((w[:, :-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:, :-1] + 1)**2), axis=1)
    
        return term1 + sum_term + term3
    
        
class Ackley:
    def __init__(self, dims=3, turn = 0.1):
        self.dims    = dims
        self.lb      = -5 * np.ones(dims)
        self.ub      =  5 * np.ones(dims)
        self.counter = 0
        self.turn    = turn

    def __call__(self, x):
        x = np.array(x / self.turn).round(0) * self.turn
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        result = (-20*np.exp(-0.2 * np.sqrt(np.inner(x,x) / x.size )) -np.exp(np.cos(2*np.pi*x).sum() /x.size) + 20 +np.e )
        return result    
    
    def multi(self,X, a=20, b=0.2, c=2*np.pi):
        X = np.array(X / self.turn).round(0) * self.turn
        d = X.shape[1]
        sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(X**2, axis=1) / d))
        cos_term = -np.exp(np.sum(np.cos(c * X), axis=1) / d)
        result = sum_sq_term + cos_term + a + np.exp(1)
        return result
        
    
class Rastrigin:
    def __init__(self, dims=3, turn = 0.1):
        self.dims   = dims
        self.lb    = -5 * np.ones(dims)
        self.ub    =  5 * np.ones(dims)
        self.counter = 0
        self.turn    = turn

    def __call__(self, x, A=10):
        x = np.array(x / self.turn).round(0) * self.turn
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        n = len(x)
        sum = np.sum(x**2 - A * np.cos(2 * np.pi * x))
        result = A*n + sum
        return result 
    
    def multi(self,X, A=10):
        X = np.array(X / self.turn).round(0) * self.turn
        return A * X.shape[1] + (X**2 - A * np.cos(2 * np.pi * X)).sum(axis=1)
    
class Rosenbrock:
    def __init__(self, dims=3, turn = 0.1):
        self.dims   = dims
        self.lb    = -5 * np.ones(dims)
        self.ub    =  5 * np.ones(dims)
        self.counter = 0
        self.turn    = turn

    def __call__(self, x):
        x = np.array(x / self.turn).round(0) * self.turn
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        result = np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
        return result 
    
    def multi(self,X, a=1, b=100):
        X = np.array(X / self.turn).round(0) * self.turn
        return ((a - X[:, :-1])**2 + b * (X[:, 1:] - X[:, :-1]**2)**2).sum(axis=1)

class Michalewicz:
    def __init__(self, dims=3, turn = 0.0001):
        self.dims   = dims
        self.lb    =  0 * np.ones(dims)
        self.ub    =  np.pi * np.ones(dims)
        self.counter = 0
        self.turn    = turn

    def __call__(self, x, m=10):
        x = np.array(x / self.turn).round(0) * self.turn
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        d = len(x)
        total = 0
        for i in range(d):
            total += np.sin(x[i]) * np.sin((i + 1) * x[i]**2 / np.pi)**(2 * m)
        # return -total
        # return 1/total
        return -total + d - 0.3
    
    def multi(self,X, m=10):
        X = np.array(X / self.turn).round(0) * self.turn
        i = np.arange(1, X.shape[1] + 1)
        return -np.sum(np.sin(X) * np.sin(i * X**2 / np.pi)**(2 * m), axis=1) + self.dims - 0.3

class Schwefel:
    def __init__(self, dims=3, turn = 1):
        self.dims   = dims
        self.lb    =  -500 * np.ones(dims)
        self.ub    =  500 * np.ones(dims)
        self.counter = 0
        self.turn    = turn

    def __call__(self, x):
        x = np.array(x / self.turn).round(0) * self.turn
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        dimension = len(x)
        sum_part = np.sum(-x * np.sin(np.sqrt(np.abs(x))))
        if np.all(np.array(x) == 421, axis = 0):
            return 0
        return 418.9829 * dimension + sum_part
    
    def multi(self,X):
        X = np.array(X / self.turn).round(0) * self.turn
        return 418.9829 * X.shape[1] - (X * np.sin(np.sqrt(np.abs(X)))).sum(axis=1)


class Griewank:
    def __init__(self, dims=3,turn = 1):
        self.dims   = dims
        self.lb    =  -600 * np.ones(dims)
        self.ub    =  600 * np.ones(dims)
        self.counter = 0
        self.turn    = turn

    def __call__(self, x):
        x = np.array(x / self.turn).round(0) * self.turn
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        # sum_term = np.sum(x ** 2)
        # prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        # return 1 + sum_term / 4000 - prod_term
        sum_term = np.sum(x**2) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return sum_term - prod_term + 1
    
    def multi(self,X):
        X = np.array(X / self.turn).round(0) * self.turn
        sum_term = (X**2).sum(axis=1) / 4000
        prod_term = np.prod(np.cos(X / np.sqrt(np.arange(1, X.shape[1] + 1))), axis=1)
        return sum_term - prod_term + 1








class Surrogate:
    def __init__(self, dims=3, name = 'method', f=None, iters = None):
        self.dims  = dims
        self.name  = name
        self.f     = f
        self.counter = 0
        self.tracker = tracker(name+str(dims))
        self.iters = iters
        self.turn = f.turn
        

    def __call__(self, x):
        x = np.array(x)
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        result = self.f(x.reshape(self.dims))
        
        self.tracker.track( result, x, self.iters)
                
        return result 
    
    
    
    
    
    
    
    
    
