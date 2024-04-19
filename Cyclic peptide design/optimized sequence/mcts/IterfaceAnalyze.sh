#!/bin/bash


# Loop through all .pdb files in the current directory
for pdb_file in *.pdb; do
    # Extract the base name without the .pdb extension
    # if [[ $pdb_file == *_min.pdb ]]; then
    #     echo "Skipping already minimized file $pdb_file"
    #     continue # Skip to the next file
    # fi
    
    pdb_name="${pdb_file%.pdb}"

    echo "Processing $pdb_file..."

    if [[ -e "${pdb_name}_min.pdb" ]]; then
        echo "The file ${pdb_name}_min.pdb already exists, skipping minimization."
        continue # Skip to the next pdb file
    fi

    # Run the minimization command
    minimize.default.linuxgccrelease -s "${pdb_file}" -run:min_tolerance 0.001 -use_truncated_termini -relax:bb_move false

    # Check if the file {pdb_name}_0001.pdb was created and rename it
    if [[ -e "${pdb_name}_0001.pdb" ]]; then
        mv "${pdb_name}_0001.pdb" "${pdb_name}_min.pdb"
    else
        echo "Expected file ${pdb_name}_0001.pdb does not exist."
        continue # Skip to the next pdb file
    fi

    # Remove the score.sc file if it exists
    if [[ -e score.sc ]]; then
        rm score.sc
    fi

    # Run the InterfaceAnalyzer command
    InterfaceAnalyzer.default.linuxgccrelease -s "${pdb_name}_min.pdb" @pack_input_options.txt > "${pdb_name}_log.txt"

    # Rename the pack_input_score.sc if it exists
    if [[ -e pack_input_score.sc ]]; then
        mv pack_input_score.sc "${pdb_name}_pack_input_score.sc"
    else
        echo "Expected file pack_input_score.sc does not exist."
    fi

    echo "Finished processing $pdb_file"
done

echo "All PDB files processed."

