import numpy as np
import json
import os

############################### Define the functions ###############################
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
        return result, 100/(result+0.01)
    
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
        return result, -result
    
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
        return result, 100/(result/(self.dims*100)+0.01)

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
        sum_term = np.sum(x ** 2)
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        result = 1 + sum_term / 4000 - prod_term
        return result, 10/(result/(self.dims)+0.001)



class Michalewicz:
    def __init__(self, dims=3, turn = 0.01):
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
        return -total, total
        # return 1/total
        # result = -total + d - 0.3
        # return -total + d - 0.3

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
            return 0, 10000
        result = 418.9829 * dimension + sum_part
        # return result, 100/(result/(self.dims * 100)+0.01)
        return result, -result/100



class Levy:
    def __init__(self, dims=1, turn = 0.1):
        self.dims    = dims
        self.lb      = -10 * np.ones(dims)
        self.ub      =  10 * np.ones(dims)
        self.counter = 0
        self.turn    = turn
        self.round   = 1

    def __call__(self, x):
        x = np.array(x / self.turn).round(0) * self.turn
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        w = []
        for idx in range(0, len(x)):
            w.append( 1 + (x[idx] - 1) / 4 )
        w = np.array(w)
        
        term1 = ( np.sin( np.pi*w[0] ) )**2;
        
        term3 = ( w[-1] - 1 )**2 * ( 1 + ( np.sin( 2 * np.pi * w[-1] ) )**2 );

        term2 = 0;
        for idx in range(1, len(w) ):
            wi  = w[idx]
            new = (wi-1)**2 * ( 1 + 10 * ( np.sin( np.pi* wi + 1 ) )**2)
            term2 = term2 + new
        
        result = term1 + term2 + term3
        return result, -result
    


class tracker:
    def __init__(self, foldername):
        self.counter   = 0
        self.results   = []
        self.x         = []
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
        np.save(self.foldername +'/result.npy',np.array(self.results),allow_pickle=True)
            
    def track(self, result, x = None, saver = False):
        self.counter += 1
        if result < self.curt_best:
            self.curt_best = result
            self.curt_best_x = x
        print("")
        print("="*10)
        print("#samples:", self.counter, "total samples:", len(self.results)+1)
        print("="*10)
        print("current best f(x):", self.curt_best)
        print("current best x:", np.around(self.curt_best_x, decimals=4))
        self.results.append(self.curt_best)
        self.x.append(x)
        if saver == True:
            self.dump_trace()
        if self.counter % 20 == 0:
            self.dump_trace()
        if round(self.curt_best,5) == 0:
            self.dump_trace()



class Surrogate:
    def __init__(self, dims=3, name = 'method', f=None, iters = None):
        self.dims  = dims
        self.name  = name
        self.f     = f
        self.counter = 0
        self.tracker = tracker(name+str(dims))
        self.iters = iters
        self.turn = f.turn
        

    def __call__(self, x, saver = False):
        x = np.array(x)
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        result, result2 = self.f(x.reshape(self.dims))
        
        self.tracker.track( result, x, saver)
                
        return result, result2
    
    
    
    
    
    
    
    
    
