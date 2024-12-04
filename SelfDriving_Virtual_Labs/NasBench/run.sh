# self-driving virtual lab for NasBench
source $HOME/.bashrc

conda activate tf_25 #your environment

python3 run.py --samples 200 --method random  --random_seed 44
