import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import py4DSTEM
# import seaborn as sns
from bayes_opt import BayesianOptimization

print(py4DSTEM.__version__)

n_dim = 8
init_points=20
n_iter=4000

dataset =  py4DSTEM.read("ptycho_Si-110_18nm.h5")
dataset.calibration

def oracle(x):
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
    plt.close()
    ms_ptycho_18nm = ms_ptycho_18nm.reconstruct(
        reset=True,
        store_iterations=True,
        max_iter = round(max_iter),
        identical_slices_iter= round(identical_slices_iter),
        step_size=step_size,
    ).visualize(
        iterations_grid = 'auto',
    )
   
    plt.close()
    print(ms_ptycho_18nm.error)
    return ms_ptycho_18nm.error

def value_cal(x):
    value=oracle(x)
    return 1/value , value

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


def train_model(x1,
                x2,
                x3, 
                x4,
                x5, 
                x6, 
                x7, 
                x8,
                # x9,
                # x10,
                # x11,
                # x12,
                ):
    params = {
        "x1": round(x1,1),
        'x2': round(x2,0),
        'x3': round(x3,-3),
        'x4': round(x4,0),
        'x5': round(x5,2),
        'x6': round(x6,0),
        'x7': round(x7,1),
        'x8': round(x8,0),
        # 'x8': round(x8,1),
        # 'x9': round(x9,1),
        # 'x10': round(x10,1),
        # 'x11': round(x11,1),
        # 'x12': round(x12,1),
                  }
    print(params)

    x_test = [params['x1']
              ,params['x2']
              ,params['x3']
              ,params['x4']
              ,params['x5']
              ,params['x6']
              ,params['x7']
              ,params['x8']]
    
              # ,params['x7'],params['x8'],params['x9'],params['x10'],params['x11'],params['x12'],]
    x_test=np.array(x_test)

    score = np.array(value_cal(x_test))
    return score[0]


bounds = {
          'x1': (1, 30),
          'x2': (1, 200),
          'x3': (1e3, 300e3),
          'x4': (1, 20.49),
          'x5': (0.01, 1),
          'x6': (1, 500),
          'x7': (1, 50),
          'x8': (1, 100),
          # 'x8': (0.1, 0.8),
          # 'x9': (0.1, 0.8), 
          # 'x10': (0.1, 0.8),
          # 'x11': (0.1, 0.8),
          # 'x12':(0.1,0.8),
          
          }
optimizer = BayesianOptimization(
    f=train_model,
    pbounds=bounds,
    random_state=8,
)
optimizer.maximize(init_points=init_points, n_iter=n_iter)

optimizer.max

# print(optimizer.max)

target_all=[]
params_all=[]
for i, res in enumerate(optimizer.res):
    target=res['target']
    target_all.append(target)
    
    params=[]
    for n in range(1,n_dim+1):                                 
        temp=res['params']["x%d"%(n)]
        params.append(temp)
    params=np.array(params)
    ###################
    params[0]=round(params[0],1)
    params[1]=round(params[1],0)
    params[2]=round(params[2],-3)
    params[3]=round(params[3],0)
    params[4]=round(params[4],2)
    params[5]=round(params[5],0)
    params[6]=round(params[6],1)
    params[7]=round(params[7],0)
    ########################
    # params=params.round(1)
    params_all.append(params)
    
target_all=np.array(target_all)
params_all=np.array(params_all)
target_all=1/target_all

df2 = pd.DataFrame(np.concatenate((params_all,target_all.reshape(-1,1)),axis=1))
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
df2.to_csv('results-BO.csv')

max_input = params_all[np.argmin(target_all)]
print(np.argmin(target_all))
oracle_show(max_input)
