import os
import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import numpy as np
import warnings
warnings.filterwarnings("ignore")

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
                if line.split(':')[0].strip()[-3:] == 'min' and key == line.split(':')[1].strip():
                    # Extract the value after the colon and strip whitespace
                    value = line.split(':')[-1][:-1].strip()
                    # Save the extracted value
                    values_of_interest[key] = value
                    break  # Move to the next line once the value is found

    return values_of_interest

def process_directory(directory_path):
    # List to hold all the results
    all_results = []
    
    # Loop through all the files in the directory
    for filename in os.listdir(directory_path):
        if filename.split('.')[0][-3:] == 'log':  # Make sure to process only .txt files
            file_path = os.path.join(directory_path, filename)
            extracted_values = extract_values_from_rosetta_output(file_path)
            extracted_values['File'] = filename  # Add filename to the results
            extracted_values['seq'] = parser_pdb(os.path.join(directory_path, filename.split('.')[0][:-4] + '_min.pdb'), 'B')
            extracted_values.update(extract_from_score(os.path.join(directory_path, filename.split('.')[0][:-4] + '_pack_input_score.sc')))
            all_results.append(extracted_values)
    
    # Convert the list of results to a DataFrame
    df = pd.DataFrame(all_results)
    return df

def parser_pdb(pdb_file: str, chain_id: str):
    parser = PDBParser()
    id = pdb_file.split('.')[0]
    structure = parser.get_structure(id, pdb_file)
    for chain in structure.get_chains():
        if chain.id == chain_id:
            chain_residue = []
            for residue in chain.get_residues():
                chain_residue.append(seq1(residue.get_resname()))
    return ''.join(chain_residue)

def extract_from_score(file_path):
    score = {}
    with open(file_path, 'r') as file:
        content = file.readlines()
    keys = content[1].split()
    values = content[2].split()
    for i in range(1, len(content[1].split()) - 1):
        score[keys[i]] = values[i]
    return score


# Usage
directory_path = './'  # Replace with the path to your directory
df = process_directory(directory_path)
df.to_csv('./interface.csv', index=None)
