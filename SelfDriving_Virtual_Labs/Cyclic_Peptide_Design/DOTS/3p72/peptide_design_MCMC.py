# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + colab={"base_uri": "https://localhost:8080/", "height": 430} id="T1rifox3d3As" outputId="df3f70b6-0c00-448a-ba29-1b9e8ac3e736"
import numpy as np
import random

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import time

import os
import random
from tqdm import tqdm

from abc import ABC, abstractmethod
from collections import defaultdict
import math

from collections import namedtuple

from get_sc import *
# -

# # Set the random seed for reproducibility
# random.seed()

pdb = '3p72.pdb'
seq_len = 11

seq_folder = './result/MCMC_peptide_' + pdb[:4] + '_' + time.strftime("%Y-%m-%d_%H-%M", time.localtime())
if not os.path.exists(seq_folder):
# If it doesn't exist, create it
    os.makedirs(seq_folder)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_nature_value(pdb_file):
    pdb_name = pdb_file.split('.')[0]
    def run_command_silently(command):
        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    run_command_silently(f"InterfaceAnalyzer.default.linuxgccrelease -s {pdb_name}.pdb @pack_input_options.txt > {pdb_name}_log.txt")
    if os.path.exists("pack_input_score.sc"):
        os.rename("pack_input_score.sc", f"{pdb_name}_pack_input_score.sc")
    else:
        print("Expected file pack_input_score.sc does not exist.")
    run_command_silently(f"mv {pdb_name}_* ./{seq_folder}")
    return extract_values_from_rosetta_output(f'./{seq_folder}/{pdb_name}_log.txt')

nature_value = get_nature_value(pdb)
print(nature_value)
print('nature target: ', float(nature_value['SHAPE COMPLEMENTARITY VALUE']) * float(nature_value['INTERFACE DELTA SASA']) / 100)
print('TRY TO EXCEED IT!!!!!')

def mutate(seq, plddt):
    flip = random.randint(0,6)
    tup = seq
    index = random.randint(0,seq_len-1)
    if flip <= 0:
        while True:
            mutation = random.randint(0,19)
            if mutation != tup[index]:
                break
        tup[index] = mutation
    elif flip <= 1:
        for i in range(int(seq_len/3)):
            index_2 = random.randint(0, len(tup)-1)
            tup[index_2] = random.randint(0,19)
    elif flip <= 5:
        p = np.maximum(1-np.array(plddt), 0) * 100
        p = softmax(p)
        index_2 = np.random.choice(seq_len, np.random.randint(0, seq_len) + 1, p=p, replace=False)
        for i in index_2:
            tup[i] = random.randint(0,19)
    elif flip:
        while True:
            new_tup = [random.randint(0, 19) for _ in range(seq_len)]
            if new_tup != tup:
                break
            tup = new_tup
    return tup

def _design_mcmc(init_seq, steps=1000, half_life=200, T_init=0.01, mutation_rate=1,
                   seq_logits=None, save_best=True, **kwargs):
    '''
    MCMC with simulated annealing
    ----------------------------------------
    steps = number for steps for the MCMC trajectory
    half_life = half-life for the temperature decay during simulated annealing
    T_init = starting temperature for simulated annealing. Temperature is decayed exponentially
    mutation_rate = number of mutations at each MCMC step
    '''

    # code borrowed from: github.com/bwicky/oligomer_hallucination

    # gather settings
    # initialize
    plddt, best_target, current_target = None, 0, np.inf

    # run!
    print("Running MCMC with simulated annealing...")
    print(f'init_seq:{init_seq}')
    for i in range(steps):

      # update temperature
      T = T_init * (np.exp(np.log(0.5) / half_life) ** i) 

      # mutate sequence
      if i == 0:
        mut_seq = init_seq
      else:
        mut_seq = mutate(seq=current_seq, plddt=plddt)
      print(f'mut_seq:{mut_seq}')

      # get loss
      values, aux = get_value(seq_folder, mut_seq, pdb='3p72.pdb')
      target = float(values['SHAPE COMPLEMENTARITY VALUE']) * float(values['INTERFACE DELTA SASA']) / 100
      plddt = aux["all"]["plddt"].mean(0)
      plddt = plddt[-seq_len:]
      # decide
      delta = target - current_target
      if i == 0 or delta > 0 or np.random.uniform() < np.exp( delta / T):
        print("*" * 100)
        print(f'{i}: Current = {target}, Current best = {best_target}')
        # accept
        (current_seq,current_target) = (mut_seq,target)
        
        if target > best_target:
          best_target = target

seq_init = [random.randint(0, 19) for _ in range(seq_len)]

_design_mcmc(seq_init)
