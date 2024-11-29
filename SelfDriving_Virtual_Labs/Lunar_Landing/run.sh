source $HOME/.bashrc

for((i=1;i<=5;i++));
do
python3 run.py --func LunarLander --dims 100 --iterations 10000 --method DANTE
done

