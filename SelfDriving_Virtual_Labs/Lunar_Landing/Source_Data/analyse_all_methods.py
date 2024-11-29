import os
import numpy as np
import matplotlib.pyplot as plt

def process(name):
    f = open('./'+name+'/result')
    yourList = f.readlines()
    yourList2=[]
    max_len = 0
    for i in yourList:
        i=i.strip('[')
        i=i.strip(']\n')
        i = [item.strip() for item in i.split(',')]
        yourList2.append(i)
        if len(i) > max_len:
            max_len = len(i)
    yourList3 = []
    samples = []
    for i in yourList2:
        ii = np.array(i).astype(float)
        samples.append(len(ii))
        if len(ii) < max_len:
            ii = np.concatenate((ii, np.zeros(max_len-len(ii))))
        yourList3.append(ii)
    yourList3 = np.array(yourList3)
    mean = np.mean(yourList3, axis=0)
    std = np.std(yourList3, axis=0)
    print(name,len(yourList2))
    return mean, std



labels = ['Random',  'DOO',     'SOO',   'VOO',    'Shiwa',  'CMA-ES', 'DifferentialEvolution', 'DualAnnealing','MCMC','DANTE']
labels1 = ['Random', 'DOO',    'SOO',    'VOO',    'Shiwa',  'CMA-ES', 'Diff-Evo'              , 'DualAnnealing','MCMC','DANTE']
colors  = ["#69B77F","#F0AC42",'#B8C7E6','#1F77B4','#9467BD','#8C564B','#E377C2',                '#7F7F7F',   '#2CA02C', '#FAB49B', ]



funcs = 'LunarLander'
dims = 100
plt.figure()
for l,i in enumerate(labels):
    try:
        mean, std = process(i + '-' + funcs + str(dims))
        plt.plot(np.arange(len(mean)), mean, '-', label = labels1[l], color = colors[l])
        plt.fill_between(np.arange(len(mean)), mean -std, mean + std, alpha=0.2, facecolor=colors[l])
    except:
        pass

plt.xlabel('Number of data acquisition')
plt.ylabel('f(x)')
plt.title(funcs +'-'+str(dims)+'d')


plt.legend(prop={'size':10})
plt.ylim(-300,180)
plt.tight_layout()
plt.savefig(f'./test-{funcs}.png')










