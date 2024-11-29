import numpy as np
import json
import os
# import gym
import gymnasium as gym


############################### Define the functions ###############################
class LunarLander:
    
    def __init__(self, dims=100, turn=1):
        self.dims         = dims
        self.lb           = 0 * np.ones(self.dims)
        self.ub           = 3 * np.ones(self.dims)
        self.counter      = 0
        self.env          = gym.make("LunarLander-v3")
        
        self.turn    = turn
        
   
    def __call__(self, x):
        x = np.array(x / self.turn).round(0) * self.turn
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        
        obs, info    = self.env.reset(seed=42)
        # print(obs)
        done1   = False
        done2   = False
        totalr = 0.
        steps  = 0
        
        for i in range(self.dims):
            action = round(x[i])
            # print(action,'action')
            obs, r, done1, done2, _ = self.env.step(action)
            totalr += r
            steps  += 1
            if done1 or done2:
                break

        return totalr




class tracker:
    def __init__(self, foldername):
        self.counter   = 0
        self.results   = []
        self.curt_best = float("-inf")
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
        if result > self.curt_best:
            self.curt_best = result
            self.curt_best_x = x
        print("")
        print("="*10)
        print("iteration:", self.counter, "total samples:", len(self.results) )
        print("="*10)
        print("current best f(x):", self.curt_best)
        print("current best x:", np.around(self.curt_best_x, decimals=0))
        self.results.append(self.curt_best)
        self.counter += 1
        if len(self.results) == iters:
            self.dump_trace()
        elif round(self.curt_best,5) == 0:
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
        self.lb   = f.lb
        self.ub   = f.ub

    def __call__(self, x):
        x = np.array(x)
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        result = self.f(x.reshape(self.dims))
        
        self.tracker.track( result, x, self.iters)
                
        return result 
    
    
    
    

