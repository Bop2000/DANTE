#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/ohuelab/ColabDesign-cyclic-binder/blob/cyc_binder/cyclic_peptide_binder_design.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# This notebook is based on the AfDesign binder hallucination protocol in ColabDesign, which is published by Dr. Sergey Ovchinnikov on [GitHub](https://github.com/sokrypton/ColabDesign/tree/main/af)

# # AfDesign - cyclic peptide binder design
# For a given protein target and cyclic binder length, generate/hallucinate a cyclic binder sequence AlphaFold thinks will bind to the target structure. To do this, we maximize number of contacts at the interface and maximize pLDDT of the binder.

# In[1]:


#@title **setup**
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.shared.utils import copy_dict
from colabdesign.af.alphafold.common import residue_constants

from IPython.display import HTML
import numpy as np
import matplotlib.pyplot as plt

#########################
  
def add_cyclic_offset(self, bug_fix=True):
  '''add cyclic offset to connect N and C term'''
  def cyclic_offset(L):
    i = np.arange(L)
    ij = np.stack([i,i+L],-1)
    offset = i[:,None] - i[None,:]
    c_offset = np.abs(ij[:,None,:,None] - ij[None,:,None,:]).min((2,3))
    if bug_fix:
      a = c_offset < np.abs(offset)
      c_offset[a] = -c_offset[a]
    return c_offset * np.sign(offset)
  idx = self._inputs["residue_index"]
  offset = np.array(idx[:,None] - idx[None,:])

  if self.protocol == "binder":
    c_offset = cyclic_offset(self._binder_len)
    offset[self._target_len:,self._target_len:] = c_offset
  self._inputs["offset"] = offset


pdb_list = ['5h5q.pdb', '5tu6.pdb', '6d40.pdb', '6u6k.pdb', '6vxy.pdb', '7ezw.pdb', '7k2j.pdb']

# In[2]:

for pdb_name in pdb_list:
  print(pdb_name)
  for num in range(3):
    #@title **prep inputs**
    import re
    #@markdown ---
    #@markdown **target info**
    pdb = f"../pdb/{pdb_name}" #@param {type:"string"}
    #@markdown - enter PDB code or UniProt code (to fetch AlphaFoldDB model) or leave blink to upload your own
    target_chain = "A" #@param {type:"string"}
    if pdb == '../pdb/3zgc.pdb':
        target_hotspot = "334,363,364,380,382,415,483,508,525,530,555,556,572,574" #@param {type:"string"}
        binder_len = 7
    elif pdb == '../pdb/3p72.pdb':
        target_hotspot = "128,130,152,230,234,235,236"
        binder_len = 11
    elif pdb == '../pdb/1sfi.pdb':
        target_hotspot = "24,40,174,193,194,79"
        binder_len = 14
    elif pdb == '../pdb/5h5q.pdb':
        target_hotspot = "33,43,45,53,142"
        binder_len = 13
    elif pdb == '../pdb/5tu6.pdb':
        target_hotspot = "69,71,118,119,220,269,271"
        binder_len = 7
    elif pdb == '../pdb/1smf.pdb':
        target_hotspot = "174,25,41,79,193,177,173,40"
        binder_len = 14
    elif pdb == '../pdb/4ib5.pdb':
        target_hotspot = "36,37,39-42,52,54,57,67,69,71,101,103,106,108,110,112"
        binder_len = 13
    elif pdb == '../pdb/6u6k.pdb':
        target_hotspot = "81,82,91-94,143-146,149"
        binder_len = 11
    elif pdb == '../pdb/7ezw.pdb':
        target_hotspot = "46-48,52,56,60,88,90-92,94,100-103,112,153,155,157,166"
        binder_len = 11
    elif pdb == '../pdb/7k2j.pdb':
        target_hotspot = "334,363,364,380,382,414,415,483,508,509,525,530,555,556,572,577,602,603"
        binder_len = 7
    elif pdb == '../pdb/4kel.pdb':
        target_hotspot = "195,194,193,192,174,175,25,83,41"
        binder_len = 14
    elif pdb == '../pdb/1sld.pdb':
        target_hotspot = "25,84,110"
        binder_len = 6
    elif pdb == '../pdb/6vxy.pdb':
        target_hotspot = "195,193,174,41,25"
        binder_len = 14
    elif pdb == '../pdb/6d40.pdb':
        target_hotspot = "603,719,738,761-763"
        binder_len = 14
    else:
        raise NotImplementedError
    if target_hotspot == "": target_hotspot = None
    #@markdown - restrict loss to predefined positions on target (eg. "1-10,12,15")
    target_flexible = False #@param {type:"boolean"}
    #@markdown - allow backbone of target structure to be flexible

    #@markdown ---
    #@markdown **binder info**
    cyclic_offset = True #@param {type:"boolean"}
    #@markdown - if True, use cyclic petide complex offset for hallucination of cyclic peptides
    bugfix = True #@param {type:"boolean"}
    #@markdown - if True, use bug fiexed version for cyclic offset
    #@param {type:"integer"}
    #@markdown - length of binder to hallucination
    binder_seq = "" #@param {type:"string"}
    binder_seq = re.sub("[^A-Z]", "", binder_seq.upper())
    if len(binder_seq) > 0:
      binder_len = len(binder_seq)
    else:
      binder_seq = None
    #@markdown - if defined, will initialize design with this sequence

    #@markdown ---
    #@markdown **model config**
    use_multimer = True #@param {type:"boolean"}
    #@markdown - use alphafold-multimer for design
    num_recycles = 6 #@param ["0", "1", "3", "6"] {type:"raw"}
    num_models = "1" #@param ["1", "2", "3", "4", "5", "all"]
    num_models = 5 if num_models == "all" else int(num_models)
    #@markdown - number of trained models to use during optimization

    x = {"pdb_filename":pdb,
        "chain":target_chain,
        "binder_len":binder_len,
        "hotspot":target_hotspot,
        "use_multimer":use_multimer,
        "rm_target_seq":target_flexible}
          

    if "x_prev" not in dir() or x != x_prev:
      clear_mem()
      model = mk_afdesign_model(
        protocol="binder",
        use_multimer=x["use_multimer"],
        num_recycles=num_recycles,
        recycle_mode="sample",
        data_dir='/home/aih/yangtao.chen/tools/alphafold_params/'
      )
      model.prep_inputs(
        **x,
        ignore_missing=False
      )
      x_prev = copy_dict(x)
      print("target length:", model._target_len)
      print("binder length:", model._binder_len)
      binder_len = model._binder_len

    # Set cyclic offset
    if cyclic_offset:
      if bugfix:
          print("Set bug fixed cyclic peptide complex offset. The cyclic peptide binder will be hallucionated.")
          add_cyclic_offset(model, bug_fix=True)
      else:
          print("Set not bug fixed cyclic peptide complex offset. The cyclic peptide binder will be hallucionated.")
          add_cyclic_offset(model, bug_fix=False)
    else:
      print("Don't set cyclic offset. The linear peptide binder will be hallucionated.")


    # In[3]:


    #@title **cyclic peptide complex offset visualization**
    # cyclic peptide complex offset
    offset = model._inputs["offset"]
    offset[model._target_len:, :model._target_len] = 0
    offset[:model._target_len, model._target_len:] = 0
    plt.figure()
    plt.title("cyclic peptide complex offset")
    plt.imshow(offset, cmap="bwr_r", vmin=-model._target_len-5, vmax=model._target_len+5)
    # cyclic peptide complex offset (cyclic peptide only)
    plt.figure()
    plt.title("cyclic peptide complex offset (cyclic peptide only)")
    for i in range(model._binder_len):
        for j in range(model._binder_len):
            plt.text(j, i, str(offset[i][j]), va='center', ha='center', fontsize=8)
    plt.imshow(offset[-model._binder_len:, -model._binder_len:], cmap="bwr_r", vmin=-model._binder_len-5, vmax=model._binder_len+5)


    # In[4]:


    #@title **run AfDesign**
    from scipy.special import softmax

    optimizer = "pssm_semigreedy" #@param ["pssm_semigreedy", "3stage", "semigreedy", "pssm", "logits", "soft", "hard"]
    #@markdown - `pssm_semigreedy` - uses the designed PSSM to bias semigreedy opt. (Recommended)
    #@markdown - `3stage` - gradient based optimization (GD) (logits → soft → hard)
    #@markdown - `pssm` - GD optimize (logits → soft) to get a sequence profile (PSSM).
    #@markdown - `semigreedy` - tries X random mutations, accepts those that decrease loss
    #@markdown - `logits` - GD optimize logits inputs (continious)
    #@markdown - `soft` - GD optimize softmax(logits) inputs (probabilities)
    #@markdown - `hard` - GD optimize one_hot(logits) inputs (discrete)

    #@markdown WARNING: The output sequence from `pssm`,`logits`,`soft` is not one_hot. To get a valid sequence use the other optimizers, or redesign the output backbone with another protocol like ProteinMPNN.

    #@markdown ----
    #@markdown #### advanced GD settings
    GD_method = "sgd" #@param ["adabelief", "adafactor", "adagrad", "adam", "adamw", "fromage", "lamb", "lars", "noisy_sgd", "dpsgd", "radam", "rmsprop", "sgd", "sm3", "yogi"]
    learning_rate = 0.1 #@param {type:"raw"}
    norm_seq_grad = True #@param {type:"boolean"}
    dropout = True #@param {type:"boolean"}

    model.restart(seq=binder_seq)
    model.set_optimizer(optimizer=GD_method,
                        learning_rate=learning_rate,
                        norm_seq_grad=norm_seq_grad)
    models = model._model_names[:num_models]

    flags = {"num_recycles":num_recycles,
            "models":models,
            "dropout":dropout}

    if optimizer == "3stage":
      model.design_3stage(120, 60, 10, **flags)
      pssm = softmax(model._tmp["seq_logits"],-1)

    if optimizer == "pssm_semigreedy":
      model.design_pssm_semigreedy(120, 32, **flags)
      pssm = softmax(model._tmp["seq_logits"],1)

    if optimizer == "semigreedy":
      model.design_pssm_semigreedy(0, 32, **flags)
      pssm = None

    if optimizer == "mcmc":
      model._design_mcmc()
      pssm = None

    if optimizer == "pssm":
      model.design_logits(120, e_soft=1.0, num_models=1, ramp_recycles=True, **flags)
      model.design_soft(32, num_models=1, **flags)
      flags.update({"dropout":False,"save_best":True})
      model.design_soft(10, num_models=num_models, **flags)
      pssm = softmax(model.aux["seq"]["logits"],-1)
    print(model._tmp["best"]["aux"]["log"])
    model.save_pdb(f"./{pdb_name.split('.')[0]}_mcmc_{num}.pdb")



# In[ ]:


#@title display hallucinated protein {run: "auto"}

# In[ ]:


# In[ ]:




# In[ ]:


#@markdown ### Amino acid probabilties
# In[ ]:


# log


# In[ ]:




