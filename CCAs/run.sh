# self-driving virtual lab for CCAs design

source $HOME/.bashrc

conda activate tf_25 #your environment
current_dir=$(pwd)
echo $current_dir

# DOTS BCC
dir="/DOTS-BCC"
cd "$current_dir$dir"
python3 run.py --iter 1

# DOTS FCC
dir="/DOTS-FCC"
cd "$current_dir$dir"
python3 run.py --iter 1

