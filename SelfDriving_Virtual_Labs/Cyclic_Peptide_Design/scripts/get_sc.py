# Predict the structure of protein-cyclic peptide complex and 
# calculate the rosette metrics.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.shared.utils import copy_dict
import subprocess
import numpy as np
import re

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

def int2aa(seq):
    aacode = {
        "0": "A",
        "1": "R",
        "2": "N",
        "3": "D",
        "4": "C",
        "5": "Q",
        "6": "E",
        "7": "G",
        "8": "H",
        "9": "I",
        "10": "L",
        "11": "K",
        "12": "M",
        "13": "F",
        "14": "P",
        "15": "S",
        "16": "T",
        "17": "W",
        "18": "Y",
        "19": "V" 
    }
    aa = [aacode[str(int(i))] for i in seq]
    return "".join(aa)

def set_model(binder, pdb='3zgc.pdb'):
    target_chain = "A" # Chain id of protein
    if pdb == '1sfi.pdb':
        target_hotspot = "40,41,57,97,99,175,192,195"  # define the interface hotspot
    elif pdb == '1sld.pdb':
        target_hotspot = "25,84,110"
    elif pdb == '1smf.pdb':
        target_hotspot = "39-42,57,60,94,96,99,151,189-193,195,213-216,219,210,226"
    elif pdb == '3p72.pdb':
        target_hotspot = "81,106,128,130,152,230,234,235,236"
    if pdb == '3zgc.pdb':
        target_hotspot = "334,363,364,380,382,415,483,508,525,530,555,556,572,574"
    elif pdb == '4ib5.pdb':
        target_hotspot = "36,37,39-42,52,54,57,67,69,71,101,103,106,108,110,112"
    elif pdb == '4kel.pdb':
        target_hotspot = "40,41,57,98,99,151,174,175,192,193,195,215,216,218"
    elif pdb == '5h5q.pdb':
        target_hotspot = "33,43,45,53,142"
    elif pdb == '5tu6.pdb':
        target_hotspot = "69,71,118,119,220,269,271"
    elif pdb == '6d40.pdb':
        target_hotspot = "603,719,738,761-763"
    elif pdb == '6u6k.pdb':
        target_hotspot = "81,82,91-94,143-146,149"
    elif pdb == '6vxy.pdb':
        target_hotspot = "175,192,219"
    elif pdb == '7ezw.pdb':
        target_hotspot = "46-48,52,56,60,88,90-92,94,100-103,112,153,155,157,166"
    elif pdb == '7k2j.pdb':
        target_hotspot = "334,363,364,380,382,414,415,483,508,509,525,530,555,556,572,577,602,603"
    else:
        return NotImplementedError
    if target_hotspot == "": target_hotspot = None
    target_flexible = False # allow backbone of target structure to be flexible
    
    
    binder_len = None # length of binder to hallucination
    binder_seq = binder
    binder_seq = re.sub("[^A-Z]", "", binder_seq.upper())
    if len(binder_seq) > 0:
        binder_len = len(binder_seq)
    else:
        binder_seq = None # if defined, will initialize design with this sequence
    # model config
    use_multimer = True # use alphafold-multimer for design
    num_recycles = 6 # ["0", "1", "3", "6"] 
    num_models = "1" # ["1", "2", "3", "4", "5", "all"]
    num_models = 5 if num_models == "all" else int(num_models) # number of trained models to use during optimization
    
    x = {
        "pdb_filename":pdb,
        "chain":target_chain,
        "binder_len":binder_len,
        "hotspot":target_hotspot,
        "use_multimer":use_multimer,
        "rm_target_seq":target_flexible
        }
    
    if "x_prev" not in dir() or x != x_prev:
        clear_mem()
        model = mk_afdesign_model(
        protocol="binder",
        use_multimer=x["use_multimer"],
        num_recycles=num_recycles,
        recycle_mode="sample",
        data_dir='/home/aih/yangtao.chen/tools/alphafold_params/') # Here is the dir of alphafold params

        model.prep_inputs(
        **x,
        ignore_missing=False
        )
        x_prev = copy_dict(x)
        binder_len = model._binder_len
        add_cyclic_offset(model, bug_fix=True)
        model.restart(seq=binder_seq)
    return model


def calculate(seq_folder, pdb_file):
    pdb_name = pdb_file.split('.')[0]
    
    # Define a function to run subprocess commands silently
    def run_command_silently(command):
        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    try:
        # Run the first command silently
        run_command_silently(f"minimize.default.linuxgccrelease -s {pdb_file} -run:min_tolerance 0.001 -use_truncated_termini -relax:bb_move false")
    
        # File operations
        if os.path.exists(f"{pdb_name}_0001.pdb"):
            os.rename(f"{pdb_name}_0001.pdb", f"{pdb_name}_min.pdb")
        else:
            print(f"Expected file {pdb_name}_0001.pdb does not exist.")
            exit(1)
    
        if os.path.exists("score.sc"):
            os.remove("score.sc")
    
        # Run the InterfaceAnalyzer command silently
        run_command_silently(f"InterfaceAnalyzer.default.linuxgccrelease -s {pdb_name}_min.pdb @pack_input_options.txt > {pdb_name}_log.txt")
    
        if os.path.exists("pack_input_score.sc"):
            os.rename("pack_input_score.sc", f"{pdb_name}_pack_input_score.sc")
        else:
            print("Expected file pack_input_score.sc does not exist.")
        run_command_silently(f"mv {pdb_name}* ./{seq_folder}")
    
    except subprocess.CalledProcessError:
        print("An error occurred while executing the command.")


def extract_values_from_rosetta_output(file_path):
    values_of_interest = {
        'TOTAL SASA': None,
        'NUMBER OF RESIDUES': None,
        'AVG RESIDUE ENERGY': None,
        'INTERFACE DELTA SASA': None,
        'INTERFACE HYDROPHOBIC SASA': None,
        'INTERFACE POLAR SASA': None,
        'CROSS-INTERFACE ENERGY SUMS': None,
        'SEPARATED INTERFACE ENERGY DIFFERENCE': None,
        'CROSS-INTERFACE ENERGY/INTERFACE DELTA SASA': None,
        'SEPARATED INTERFACE ENERGY/INTERFACE DELTA SASA': None,
        'DELTA UNSTAT HBONDS': None,
        'CROSS INTERFACE HBONDS': None,
        'HBOND ENERGY': None,
        'HBOND ENERGY/ SEPARATED INTERFACE ENERGY': None,
        'INTERFACE PACK STAT': None,
        'SHAPE COMPLEMENTARITY VALUE': None
    }
    
    with open(file_path, 'r') as file:
        for line in file:
            # Loop through each line in the file
            for key in values_of_interest.keys():
                # Check if the line contains the key
                try:
                    if key == line.split(':')[1].strip():
                        # Extract the value after the colon and strip whitespace
                        value = line.split(':')[-1][:-1].strip()
                        # Save the extracted value
                        values_of_interest[key] = value
                        break  # Move to the next line once the value is found
                except:
                    pass

    return values_of_interest

def extract_residue(file_path):
    residues = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                if line.split(',')[0][-5:] == 'unsat':
                    if line.split('/')[-2] != 'A':
                        numbers = re.findall(r'\d+\+', line.split('/')[-1])
                        numbers = [str(num.strip('+')) for num in numbers]
                        x = '_'.join(numbers)
                        residues.append(x)
            except:
                pass
    if residues:
        return residues[0]
    else:
        return 0



def get_value(seq_folder, array, pdb='3zgc.pdb'):
    binder_seq = int2aa(array)
    model = set_model(binder_seq, pdb)
    model.predict()
    model.save_pdb(f"{binder_seq}.pdb")
    calculate(seq_folder, f"{binder_seq}.pdb")
    values = extract_values_from_rosetta_output(f"{seq_folder}/{binder_seq}_log.txt")
    values['unsatH_residues'] = extract_residue(f"{seq_folder}/{binder_seq}_log.txt")
    return values

