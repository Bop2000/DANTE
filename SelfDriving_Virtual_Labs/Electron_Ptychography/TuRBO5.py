# !git clone https://github.com/uber-research/TuRBO/

# Commented out IPython magic to ensure Python compatibility.
# %cd TuRBO
# !pip install .
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from collections import namedtuple
from abc import ABC, abstractmethod
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import py4DSTEM
from turbo import TurboM

print(py4DSTEM.__version__)


dataset =  py4DSTEM.read("ptycho_Si-110_18nm.h5")
dataset.calibration
n_dim    = 8

def oracle_show(x):
    semiangle_cutoff = x[0] # 20
    defocus          = x[1] # 100
    # rotation_degrees = x[2] # 0
    energy           = x[2] # 200e3
    max_iter         = x[3] # 256
    step_size        = x[4] # 0.175
    identical_slices_iter = x[5] #256
    slice_thicknesses     = x[6] # 30.794230884706234
    num_slices       = x[7] # 6
    
    ms_ptycho_18nm = py4DSTEM.process.phase.MultislicePtychographicReconstruction(
        datacube=dataset,
        num_slices=round(num_slices),
        slice_thicknesses=slice_thicknesses,
        verbose=True,
        energy=energy,
        defocus=defocus,
        semiangle_cutoff=semiangle_cutoff,
        object_padding_px=(18,18),
        device='gpu',
    ).preprocess(
        plot_center_of_mass = False,
        plot_rotation=False,
    )
    # plt.close()
    ms_ptycho_18nm = ms_ptycho_18nm.reconstruct(
        reset=True,
        store_iterations=True,
        max_iter = round(max_iter),
        identical_slices_iter= round(identical_slices_iter),
        step_size=step_size,
    ).visualize(
        iterations_grid = 'auto',
    )
    
    ms_ptycho_18nm._visualize_last_iteration(
        fig=None,
        cbar=True,
        plot_convergence=True,
        plot_probe=True,
        plot_fourier_probe=True,
        padding=0,
    )
    
    print(ms_ptycho_18nm.error)
    return ms_ptycho_18nm.error
class oracle:
    def __init__(self, dim=10):
        self.dim = dim
        self.lb = np.array([1,   1,   1e3,  1,     0.01, 1,   1,    1 ])
        self.ub = np.array([30, 200, 300e3, 20.49, 1,    500, 50,   100])
        # self.lb = -5 * np.ones(dim)
        # self.ub = 10 * np.ones(dim)

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        semiangle_cutoff = x[0].round(1) # 20
        defocus          = x[1].round(0) # 100
        # rotation_degrees = x[2] # 0
        energy           = x[2].round(-3) # 200e3
        max_iter         = x[3].round(0) # 256
        step_size        = x[4].round(2) # 0.175
        identical_slices_iter = x[5].round(0) #256
        slice_thicknesses     = x[6].round(1) # 30.794230884706234
        num_slices       = x[7].round(0) # 6
        
        print(x)
        print(semiangle_cutoff, defocus, energy, step_size, identical_slices_iter, slice_thicknesses, num_slices)
        ms_ptycho_18nm = py4DSTEM.process.phase.MultislicePtychographicReconstruction(
            datacube=dataset,
            num_slices=round(num_slices),
            slice_thicknesses=slice_thicknesses,
            verbose=True,
            energy=energy,
            defocus=defocus,
            semiangle_cutoff=semiangle_cutoff,
            object_padding_px=(18,18),
            device='gpu',
        ).preprocess(
            plot_center_of_mass = False,
            plot_rotation=False,
        )
        plt.close()
        ms_ptycho_18nm = ms_ptycho_18nm.reconstruct(
            reset=True,
            store_iterations=True,
            max_iter = round(max_iter),
            identical_slices_iter= round(identical_slices_iter),
            step_size=step_size,
        )
        # .visualize(
        #     iterations_grid = 'auto',
        # )
       
        # plt.close()
        print(ms_ptycho_18nm.error)
        return ms_ptycho_18nm.error

f = oracle(n_dim)

turbo_m = TurboM(
    f=f,  # Handle to objective function
    lb=f.lb,  # Numpy array specifying lower bounds
    ub=f.ub,  # Numpy array specifying upper bounds
    n_init=20,  # Number of initial bounds from an Symmetric Latin hypercube design
    max_evals=4000,  # Maximum number of evaluations
    n_trust_regions=5,  # Number of trust regions
    batch_size=10,  # How large batch size TuRBO uses
    verbose=True,  # Print information from each batch
    use_ard=True,  # Set to true if you want to use ARD for the GP kernel
    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
    n_training_steps=50,  # Number of steps of ADAM to learn the hypers
    min_cuda=1024,  # Run on the CPU for small datasets
    device="cpu",  # "cpu" or "cuda"
    dtype="float64",  # float64 or float32
)

turbo_m.optimize()

X = turbo_m.X  # Evaluated points
fX = turbo_m.fX  # Observed values

X[:,0]=X[:,0].round(1)
X[:,1]=X[:,1].round(0)
X[:,2]=X[:,2].round(-3)
X[:,3]=X[:,3].round(0)
X[:,4]=X[:,4].round(2)
X[:,5]=X[:,5].round(0)
X[:,6]=X[:,6].round(1)
X[:,7]=X[:,7].round(0)

df2 = pd.DataFrame(np.concatenate((X,fX.reshape(-1,1)),axis=1))
df2.columns= ['semiangle_cutoff',
              'defocus',
              # 'rotation_degrees',
              'energy',
               'max_iter',
              'step_size',
              'identical_slices_iter',
              'slice_thicknesses',
              'num_slices',
              'NMSE']
df2.to_csv('results-TuRBO5.csv')

ind_best = np.argmin(fX)
f_best, x_best = fX[ind_best], X[ind_best, :]
print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, np.around(x_best, 5)))
oracle_show(x_best)
