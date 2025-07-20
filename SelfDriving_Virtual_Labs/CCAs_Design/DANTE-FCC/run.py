
"""################################################################################
> # **Introduction**
> **This notebook introduces the pipeline to design compositionally complex alloys, specifically, 6 element alloys with higher magnetic properties**

> The notebook is divided into 4 major parts :

*   **Part I** : visulization of initial data
*   **Part II** : define DANTE algorithm
*   **Part III** : define and train the CNN model
*   **Part IV** : optimization using DANTE

################################################################################

################################################################################
> # **Part - I**

*   Import initial dataset
*   Set parameters

################################################################################
"""

############################### Import libraries ###############################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense,Dropout,BatchNormalization,GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler,ModelCheckpoint
import random
from scipy import stats
from sklearn import metrics
from abc import ABC, abstractmethod
from collections import defaultdict
import math
from collections import namedtuple
from tensorflow.keras import layers
from tensorflow import keras
import argparse

parser = argparse.ArgumentParser(description='Process inputs')
parser.add_argument('--iter', type=int, help='specify the number of iteration')
args = parser.parse_args()




############################### Creat Initial Folder ###############################

#Number of iteration
round_num  = args.iter
round_name = 'Round'+str(round_num)
round_num2 = round_num-1
round_name2 = 'Round'+str(round_num2)

model_folder = "Results"
# Check if the directory exists
if not os.path.exists(model_folder):
    # If it doesn't exist, create it
    os.makedirs(model_folder)
if not os.path.exists(model_folder+'/'+round_name):
    # If it doesn't exist, create it
    os.makedirs(model_folder+'/'+round_name)
if not os.path.exists(model_folder+'/'+round_name2):
    # If it doesn't exist, create it
    os.makedirs(model_folder+'/'+round_name2)





############################### Set Paramaters ###############################

top_sample=20 # number of newly selected samples
n_model=5 # number of cnn models for predictions
n_dim=27  # Dimension of this optimization problem
rollout_round=100 #number of roullout steps for DANTE algorithm,By default, DANTE performs 100 rollout
UCT_low=False
weight = 0.2 # exploration weight = weight * max(score)
list1=[5,8,2,5,3,1,1] ##[run times, top start points, random start points, top score samples, top rank samples, top visit samples, random samples]

num_ele = 6 # number of elements
step_size = 0.5 # interval of composition
Fe_low = 2
Fe_up = 100 #Fe upper limit & lower limit
Co_low = 2
Co_up = 100 #Co upper limit & lower limit
Ni_low = 2
Ni_up = 100 #Ni upper limit & lower limit
other_low = 2
other_up = 50 #other element upper limit & lower limit
FCN_low = 60
FCN_high = 80 # sum of Fe Co Ni upper limit & lower limit




############################### Import Initial Dataset ###############################

data = pd.read_csv("data.csv")
data = data.fillna(0)
data = np.array(data)

all_input = data[:,:n_dim] # element composition as input
all_data0 = data[:,n_dim]*10   #Formation energy
all_data1 = data[:,n_dim+2]*10000 #AHC
all_data2 = data[:,n_dim+4]*100   #AHA

ef = np.where(all_data0>0.2,0,1)

print('fano factor of Formation energy:',np.var(all_data0)/np.mean(all_data0))
print('fano factor of ahc:',np.var(all_data1)/np.mean(all_data1))
print('fano factor of aha:',np.var(all_data2)/np.mean(all_data2))

if round_num  == 1:
  plt.figure()
  plt.scatter(data[:,n_dim+2],ef * data[:,n_dim+4],label='initial data')
  plt.xlabel('ahc')
  plt.ylabel('aha (ef < 0.02)')
  plt.title("aha vs ahc")
  plt.legend()
  plt.savefig(model_folder+'/'+round_name2+'/performance-comparison2.png')
else:
  plt.figure()
  plt.scatter(data[:,n_dim+2],ef * data[:,n_dim+4],label='all data')
  plt.scatter(data[:,n_dim+2][-20:],ef[-20:] * data[:,n_dim+4][-20:],label='this round')
  plt.xlabel('ahc')
  plt.ylabel('aha (ef < 0.02)')
  plt.title("aha vs ahc")
  plt.legend()
  plt.savefig(model_folder+'/'+round_name2+'/performance-comparison2.png')

################################# End of Part I ################################






"""################################################################################
> # **Part - II**

*   Define the DANTE alghorithm

################################################################################
"""

################################# DANTE alghorithm ################################

class DANTE:
    def __init__(self, exploration_weight=1):
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

    def choose(self, node):
        "Choose the best successor of node."
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        def evaluate(n):
            return n.value  # average reward
        print(f'number of visit is {self.N[node]}')
        log_N_vertex = math.log(self.N[node])
        def uct(n):
            "Upper confidence bound for trees"

            uct_value = n.value + self.exploration_weight * math.sqrt(
                log_N_vertex / (self.N[n]+1))
            return uct_value

        media_node = max(self.children[node], key=uct)

        node_rand = []
        for i in range(len(list(self.children[node]))):
            node_rand.append(list(self.children[node])[i].tup)
        node_rand = np.array(node_rand).reshape(-1,n_dim)

        if uct(media_node) > uct(node):
            print(f'media node is{media_node}')
            print(f'uct of the node is{uct(media_node)} ')
            print(f'better value media node : {media_node.value}')
            return media_node, node_rand
        return node, node_rand


    def do_rollout(self, node):
        """Make the tree one layer better. (Train for one iteration.)"""
        self._expand(node)
        self._backpropagate(path=node)

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        action = [p for p in range(0, len(node.tup))]
        self.children[node] = node.find_children(action)

    def _backpropagate(self, path):
        """Send the reward back up to the ancestors of the leaf"""
        self.N[path] += 1


class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True




############################ design the action space ############################

# composition limitations for all elements
def component_generate():
    ele_range = []
    ele_range4 = np.arange(Co_low, Co_up, step_size) #Co
    ele_range7 = np.arange(Ni_low, Ni_up, step_size) #Ni
    ele_range18 = np.arange(Fe_low, Fe_up, step_size) #Fe
    #set other elements' component
    index_other = np.setdiff1d(np.arange(n_dim), [4,7,18])
    for i in range(n_dim-3):
        exec(f"ele_range{index_other[i]} = np.arange(other_low, other_up, step_size)")

    for i in range(n_dim):
        exec(f"ele_range.append(ele_range{i})")
    ele_range = np.array(ele_range)
    return ele_range

# check the ratio of Fe/(Co+Ni)
def FCN_ratiocheck(x_input):
    x_output = np.array(x_input)
    Fe_num = x_input[18]
    Co_num = x_input[4]
    Ni_num = x_input[7]
    sum_FCN = Fe_num + Co_num + Ni_num
    ratio = 1.5
    Fe = np.round(ratio * sum_FCN / (ratio + 1),1)
    Co = np.round((sum_FCN - Fe) * (Co_num/(Co_num + Ni_num)),1)
    Ni = np.round((sum_FCN - Fe) * (Ni_num/(Co_num + Ni_num)),1)
    x_output[18] = Fe
    x_output[7] = Ni
    x_output[4] = Co
    return x_output

#check if Fe Co Ni has proper component, if not, adjust to a proper range
def sumFCN_adjust(x_input):
    if round(x_input[4],1) == 0 and round(x_input[7],1) == 0 and round(x_input[18],1) == 0:
        x_input[4] = x_input[7] = x_input[18] = 1
        x_input[np.argmax(x_input)] -=3

    x_output = np.array(x_input)
    index_FCN = [4,7,18]
    ele_exist = np.where(x_output > 0)
    index_other = np.setdiff1d(ele_exist[0], index_FCN)
    sum_FCN = x_input[4] + x_input[7] + x_input[18]
    sum_other = sum(x_input[index_other])

    if sum_FCN > FCN_high:
        percentage = FCN_high
    elif sum_FCN <FCN_low:
        percentage = FCN_low

    for i in range(3):
        x_output[index_FCN[i]] = np.round(x_output[index_FCN[i]] * percentage / sum_FCN,1)

    for i in range(len(index_other)):
        x_output[index_other[i]] = np.round(x_output[index_other[i]] * (100 - percentage) / sum_other,1)
    return x_output

# mode0: change element
def mode0(x_input,x_output,ele_exist,index_except):
    num_to_change = np.random.randint(1, 3)
    index_FCN = np.array([4,7,18])
    index_other = np.setdiff1d(ele_exist[0], index_FCN)
    index_except = np.setdiff1d(np.arange(n_dim), ele_exist[0])
    index_other_new = np.array(index_other)
    np.random.shuffle(index_other_new)
    other_to_be_added = np.array(index_except)
    np.random.shuffle(other_to_be_added)
    for i in range(num_to_change):
        x_output[other_to_be_added[i]] = x_input[index_other_new[i]]
        x_output[index_other_new[i]] = 0
    return x_output

# mode1: change component
def mode1(x_input,x_output,ele_exist,index_except):
    num_to_change = np.random.randint(1, 3)
    ele_exist_new = np.array(ele_exist[0])
    np.random.shuffle(ele_exist_new)
    for i in range(num_to_change):
        flip = np.random.randint(0,4)
        if flip == 0:
            temp = np.random.randint(1,11)
            x_output[ele_exist_new[i]] += step_size*temp
        elif flip == 1:
            temp = np.random.randint(1,11)
            x_output[ele_exist_new[i]] -= step_size*temp
            if x_output[ele_exist_new[i]] <= 0:
                x_output[ele_exist_new[i]] += 2*temp*step_size
        else:
            x_output[ele_exist_new[i]] = ele_range[ele_exist_new[i]][np.random.randint(0, len(ele_range[ele_exist_new[i]]))]
    return x_output

# prpose new CCAs
def create_new(x_input, num_ele):
    x_output = np.array(x_input).reshape(-1)
    ele_exist = np.where(x_input > 0)#index of existed elements
    index_except = np.setdiff1d(np.arange(n_dim), ele_exist[0])

    #3 modes : change element/change component/change both
    mode = np.random.randint(0,3)

    if mode == 0:
        x_output = mode0(x_input,x_output,ele_exist,index_except)

    elif mode == 1:
        x_output = mode1(x_input,x_output,ele_exist,index_except)

    elif mode == 2:
        x_output = mode0(x_input,x_output,ele_exist,index_except)
        x_output = mode1(x_input,x_output,ele_exist,index_except)

    new_exist = np.where(x_output > 0)
    num_exist_new = len(new_exist[0])
    sum_all = sum(x_output)
    for i in range(num_exist_new):
        x_output[new_exist[0][i]] = np.round(x_output[new_exist[0][i]]*100/sum_all/step_size)*step_size

    #make sure the sum is 100
    deviation = 100 - sum(x_output)
    if deviation != 0:
        ind_max = np.argmax(x_output)
        x_output[ind_max] += deviation

    #check if the new input is proper
    sum_FCN = x_output[4] + x_output[7] + x_output[18]
    if sum_FCN > FCN_high or sum_FCN < FCN_low:
        x_output = sumFCN_adjust(x_output)

    FCN_ratio = x_output[18] - 1.5*(x_output[4] + x_output[7])
    if FCN_ratio > 0:
        x_output = FCN_ratiocheck(x_output)

    x_output = ele_num_check(x_output, num_ele)
    return x_output

# check the number of elements
def ele_num_check(x_input, ele_num):
    x_output = np.array(x_input)
    x_output[np.where(x_input < 0)[0]] = 0 ########make sure every element no less than 0
    index = np.setdiff1d(np.arange(n_dim), np.array([4,7,18]))
    ele_true = np.where(x_input > 0)
    free_ele_exist = np.intersect1d(index, ele_true[0]) #index of elements existed except Fe, Co, Ni
    free_ele_except = np.setdiff1d(index, ele_true[0]) #index of elements excepted

    if len(ele_true[0]) > ele_num:
        dev = len(ele_true[0]) - ele_num #the deviation between ele_num and true ele_num
        index_deleted = np.array(free_ele_exist)
        np.random.shuffle(index_deleted)
        for i in range(dev):
            x_output[index_deleted[i]] = 0

    elif len(ele_true[0]) < ele_num:
        dev = ele_num - len(ele_true[0])
        index_added = np.array(free_ele_except)
        np.random.shuffle(index_added)
        for i in range(dev):
            x_output[index_added[i]] = step_size

    x_output = np.round(x_output*100/(sum(x_output)*step_size))*step_size
    deviation = 100 - sum(x_output)
    if deviation != 0:
        ind_max = np.argmax(x_output)
        x_output[ind_max] += deviation
    return x_output

_OT = namedtuple("opt_task", "tup value terminal")
class opt_task(_OT, Node):
    def find_children(board,action):
        if board.terminal:
            return set()
        all_tup=[]
        for index in action:
            tup = create_new(np.array(board.tup), num_ele)
            all_tup.append(tup)

        all_value = oracle(all_tup)
        is_terminal=False
        return  {opt_task(tuple(t), v, is_terminal) for t, v in  zip(all_tup,all_value)}

    def is_terminal(board):
        return board.terminal

# return the mostly visited nodes
def most_visit_node(tree_ubt, X,top_n):
  N_visit = tree_ubt.N
  childrens = [i for i in tree_ubt.children]
  children_N = []
  X_top = []
  for child in childrens:
    child_tup = np.array(child.tup)
    same = np.all(child_tup==X, axis=1)
    has_true = any(same)
    if has_true == False:
      children_N.append(N_visit[child])
      X_top.append(child_tup)
  children_N = np.array(children_N)
  X_top = np.array(X_top)
  ind = np.argpartition(children_N, -top_n)[-top_n:]
  X_topN = X_top[ind]
  return X_topN

# return random nodes
def random_node(new_x,n):
  X_rand1 = [new_x[random.randint(0, len(new_x)-1)] for i in range(n)]
  return X_rand1

print("DANTE defined!")

################################ End of Part II ################################






"""################################################################################
> # **Part - III**

*   Define and train the CNN model

################################################################################
"""

################################ Define the CNN model ################################

#########slice the data to five parts
index_random=np.arange(len(all_data2))
random.shuffle(index_random)
index_random1=index_random[:]

def model_training(X,y,name,i):
      X=X.reshape(-1,n_dim,1)
      ind=index_random[round(i*len(index_random)/5):round((1+i)*len(index_random)/5)]####1/5 data as test set
      ind2=np.setdiff1d(index_random, ind)
      X_train, X_test, y_train, y_test = X[ind2],X[ind], y[ind2],y[ind]
      model = Sequential([
            layers.Conv1D(64,kernel_size=3,strides=2,padding='same', activation='elu', input_shape=(n_dim,1)),
            layers.BatchNormalization(),
            layers.Conv1D(32,kernel_size=3,strides=2, padding='same', activation='elu'),
            layers.Conv1D(16,kernel_size=3,strides=2, padding='same', activation='elu'),
            layers.Dropout(0.2),
            layers.Conv1D(8,kernel_size=3,strides=1, padding='same', activation='elu'),
            layers.Flatten(),
            Dense(128, activation='elu'),
            Dense(1, activation='linear')
      ])
      optimizer = keras.optimizers.Adam(learning_rate=0.001)
      model.compile(optimizer=optimizer, loss='mse', metrics=["mean_squared_error"])
      model.summary()
      es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=1000)
      mc = ModelCheckpoint(model_folder+'/'+round_name+f"/{name}.h5", monitor='val_loss', mode='min', verbose=False, save_best_only=True)
      model.fit(X_train, y_train, validation_data=(X_test, y_test),  batch_size=50, epochs=5000, callbacks=[es,mc], verbose=False)
      model=keras.models.load_model(model_folder+'/'+round_name+f"/{name}.h5")
      R2,MAE=mar_r2(model,X_test,y_test)
      return model,X_test,y_test,R2,MAE

def model_training2(X,y,name,i):
      X=X.reshape(-1,n_dim,1)
      ind=index_random[round(i*len(index_random)/5):round((1+i)*len(index_random)/5)]####1/5 data as test set
      ind2=np.setdiff1d(index_random, ind)
      X_train, X_test, y_train, y_test = X[ind2],X[ind], y[ind2],y[ind]
      model = Sequential([
          layers.Conv1D(64,kernel_size=3,strides=2,padding='same', activation='elu', input_shape=(n_dim,1)),
            layers.BatchNormalization(),
            layers.Conv1D(32,kernel_size=3,strides=2, padding='same', activation='elu'),
            layers.Dropout(0.2),
            layers.Conv1D(16,kernel_size=3,strides=2, padding='same', activation='elu'),
            layers.Dropout(0.2),
            layers.Conv1D(8,kernel_size=3,strides=1, padding='same', activation='elu'),
            layers.Flatten(),
            Dense(16, activation='elu'),
          Dense(1, activation='linear')
      ])
      optimizer = keras.optimizers.Adam(learning_rate=0.002)
      model.compile(optimizer=optimizer, loss='mse', metrics=["mean_squared_error"])
      model.summary()
      es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=1000)
      mc = ModelCheckpoint(model_folder+'/'+round_name+f"/{name}.h5", monitor='val_loss', mode='min', verbose=False, save_best_only=True)
      model.fit(X_train, y_train, validation_data=(X_test, y_test),  batch_size=50, epochs=5000, callbacks=[es,mc], verbose=False)
      model=keras.models.load_model(model_folder+'/'+round_name+f"/{name}.h5")
      R2,MAE=mar_r2(model,X_test,y_test)
      return model,X_test,y_test,R2,MAE

def mar_r2(model,X_test,y_test):
    y_pred = model.predict(X_test.reshape(len(X_test),n_dim,1))
    R2=stats.pearsonr(y_pred.reshape(-1), y_test.reshape(-1))[0]
    R2=np.asarray(R2).round(5)
    MAE= metrics.mean_absolute_error(y_test.reshape(-1), y_pred.reshape(-1))
    return R2,MAE

def model_performance(model,X_test,y_test):
    perform_list=pd.read_csv(model_folder+'/'+round_name+f'/model_performance_{n_dim}d.csv')
    y_pred = model.predict(X_test.reshape(len(X_test),n_dim,1))
    R2=stats.pearsonr(y_pred.reshape(-1), y_test.reshape(-1))[0]
    R2=np.asarray(R2).round(5)
    MAE= metrics.mean_absolute_error(y_test.reshape(-1), y_pred.reshape(-1))
    y_test = pd.DataFrame(y_test)
    y_test.columns= ['ground truth']
    y_pred = pd.DataFrame(y_pred)
    y_pred.columns= ['pred']
    print("R2",R2,"MAE",MAE)
    R2MAE = pd.DataFrame([R2,MAE])
    R2MAE.columns= ['R2&MAE']
    perform_list2=pd.concat((perform_list,y_test,y_pred,R2MAE),axis=1)
    perform_list2.drop([perform_list2.columns[0]],axis=1, inplace=True)
    perform_list2.to_csv(model_folder+'/'+round_name+f'/model_performance_{n_dim}d.csv')
    return R2,MAE

print("Model defined!")

################################ Train the CNN model ################################


pd.DataFrame(np.empty(0)).to_csv(model_folder+'/'+round_name+f'/model_performance_{n_dim}d.csv')
try_lim = 5

#train the model of formation energy
for i in range(5):
    trytime=0
    model1,X_test1,y_test1,R21,MAE1 = model_training(all_input,all_data0,f'ef{i}',i)
    R20=R21
    while R21 < 0.98 and trytime<try_lim:
        trytime+=1
        model1,X_test1,y_test1,R21,MAE1 = model_training(all_input,all_data0,f'ef{i+10}',i)
        if R21>R20:
            R20=R21
            model1.save("Results/"+round_name+f'/ef{i}.h5')
    model1=keras.models.load_model(model_folder+'/'+round_name+f'/ef{i}.h5')
    R21,MAE1=model_performance(model1,X_test1,y_test1) #show and save the performance of the model

#train the model of ahc
for i in range(5):
    trytime=0
    model1,X_test1,y_test1,R21,MAE1 = model_training(all_input,all_data1,f'ahc{i}',i)
    R20=R21
    while R21 < 0.97 and trytime<try_lim:
        trytime+=1
        model1,X_test1,y_test1,R21,MAE1 = model_training(all_input,all_data1,f'ahc{i+10}',i)
        if R21>R20:
            R20=R21
            model1.save("Results/"+round_name+f'/ahc{i}.h5')
    model1=keras.models.load_model(model_folder+'/'+round_name+f'/ahc{i}.h5')
    R21,MAE1=model_performance(model1,X_test1,y_test1) #show and save the performance of the model

#train the model of aha
for i in range(5):
    trytime=0
    model2,X_test2,y_test2,R22,MAE2 = model_training2(all_input,all_data2,f'aha{i}',i)
    R20=R22
    while R22 < 0.97 and trytime<try_lim:
        trytime+=1
        model2,X_test2,y_test2,R22,MAE2 = model_training2(all_input,all_data2,f'aha{i+10}',i)
        if R22>R20:
            R20=R22
            model2.save("Results/"+round_name+f'/aha{i}.h5')
    model2=keras.models.load_model(model_folder+'/'+round_name+f'/aha{i}.h5')
    R22,MAE2=model_performance(model2,X_test2,y_test2) #show and save the performance of the model

################################ Load and use the CNN model ################################

path = os.getcwd()
name2=path+'/Results/'+round_name+'/'
models=dict()
model_list0=[]
for i in range(0,n_model):
    modelname = f'ef{i}'
    model_list0.append(modelname)
    models[modelname]= keras.models.load_model(name2+modelname+'.h5')
model_list1=[]
for i in range(0,n_model):
    modelname = f'ahc{i}'
    model_list1.append(modelname)
    models[modelname]= keras.models.load_model(name2+modelname+'.h5')
model_list2=[]
for i in range(0,n_model):
    modelname = f'aha{i}'
    model_list2.append(modelname)
    models[modelname]= keras.models.load_model(name2+modelname+'.h5')


###emsemble all models to predict
def emsemble_pred0(S,n_model=5):##formation energy
    pred_all=0
    for i in range(n_model):
        temp=models[model_list0[i]].predict(S.reshape(len(S),n_dim,1))
        pred_all+=temp
    pred_all/=n_model
    return pred_all

def emsemble_pred1(S,n_model=5):##ahc
    pred_all=0
    for i in range(n_model):
        temp=models[model_list1[i]].predict(S.reshape(len(S),n_dim,1))
        pred_all+=temp
    pred_all/=n_model
    return pred_all

def emsemble_pred2(S,n_model=5):##aha
    pred_all=0
    for i in range(n_model):
        temp=models[model_list2[i]].predict(S.reshape(len(S),n_dim,1))
        pred_all+=temp
    pred_all/=n_model
    return pred_all

def oracle(x):
    x_proposed=np.array(x).reshape(-1,n_dim,1)
    x0 = emsemble_pred0(x_proposed)
    x0 = np.where(x0>0.05,0,1)
    x1 = emsemble_pred1(x_proposed)
    x2 = emsemble_pred2(x_proposed)
    all_score = x0*x1*x2
    return all_score.reshape(len(x_proposed))

################################ model performance visualization ################################
import seaborn as sns
def model_visual1(X,y,i): # Formation energy
    model1=models[model_list0[i]]
    X=X.reshape(-1,n_dim,1)
    ind=index_random[round(i*len(index_random)/5):round((1+i)*len(index_random)/5)]####1/5 data as test set
    ind2=np.setdiff1d(index_random, ind)
    X_train, X_test, y_train, y_test = X[ind2],X[ind], y[ind2],y[ind]

    y_pred = model1.predict(X_test.reshape(len(X_test),n_dim,1))
    R2=stats.pearsonr(y_pred.reshape(-1), y_test.reshape(-1))[0]
    R2=np.asarray(R2).round(3)
    MAE= metrics.mean_absolute_error(y_test.reshape(-1), y_pred.reshape(-1))
    MAE=np.asarray(MAE).round(5)
    MAPE= metrics.mean_absolute_percentage_error(y_test.reshape(-1), y_pred.reshape(-1))
    MAPE=np.asarray(MAPE).round(5)
    plt.figure()
    sns.set()
    sns.regplot(x=y_pred, y=y_test, color='k')
    plt.title((f'Formation energy prediction by model #{i+1}: ','R2:',R2,'MAE:',MAE,'MAPE:',MAPE))
    plt.xlabel('Predicted formation energy')
    plt.xlabel('Simulated formation energy')

def model_visual2(X,y,i):# AHC
    model2=models[model_list1[i]]
    X=X.reshape(-1,n_dim,1)
    ind=index_random[round(i*len(index_random)/5):round((1+i)*len(index_random)/5)]####1/5 data as test set
    ind2=np.setdiff1d(index_random, ind)
    X_train, X_test, y_train, y_test = X[ind2],X[ind], y[ind2],y[ind]

    y_pred = model2.predict(X_test.reshape(len(X_test),n_dim,1))
    R2=stats.pearsonr(y_pred.reshape(-1), y_test.reshape(-1))[0]
    R2=np.asarray(R2).round(3)
    MAE= metrics.mean_absolute_error(y_test.reshape(-1), y_pred.reshape(-1))
    MAE=np.asarray(MAE).round(5)
    MAPE= metrics.mean_absolute_percentage_error(y_test.reshape(-1), y_pred.reshape(-1))
    MAPE=np.asarray(MAPE).round(5)
    plt.figure()
    sns.set()
    sns.regplot(x=y_pred, y=y_test, color='k')
    plt.title((f'AHC prediction by model #{i+1}: ','R2:',R2,'MAE:',MAE,'MAPE:',MAPE))
    plt.xlabel('Predicted AHC')
    plt.xlabel('Simulated AHC')

def model_visual3(X,y,i):# AHA
    model2=models[model_list2[i]]
    X=X.reshape(-1,n_dim,1)
    ind=index_random[round(i*len(index_random)/5):round((1+i)*len(index_random)/5)]####1/5 data as test set
    ind2=np.setdiff1d(index_random, ind)
    X_train, X_test, y_train, y_test = X[ind2],X[ind], y[ind2],y[ind]

    y_pred = model2.predict(X_test.reshape(len(X_test),n_dim,1))
    R2=stats.pearsonr(y_pred.reshape(-1), y_test.reshape(-1))[0]
    R2=np.asarray(R2).round(3)
    MAE= metrics.mean_absolute_error(y_test.reshape(-1), y_pred.reshape(-1))
    MAE=np.asarray(MAE).round(5)
    MAPE= metrics.mean_absolute_percentage_error(y_test.reshape(-1), y_pred.reshape(-1))
    MAPE=np.asarray(MAPE).round(5)
    plt.figure()
    sns.set()
    sns.regplot(x=y_pred, y=y_test, color='k')
    plt.title((f'AHA prediction by model #{i+1}: ','R2:',R2,'MAE:',MAE,'MAPE:',MAPE))
    plt.xlabel('Predicted AHA')
    plt.xlabel('Simulated AHA')

for i in range(5):
    model_visual1(all_input,all_data0,i)
for i in range(5):
    model_visual2(all_input,all_data1,i)
for i in range(5):
    model_visual3(all_input,all_data2,i)

################################ End of Part III ###############################






"""################################################################################
> # **Part - IV**

*   Optimization using DANTE

################################################################################

Input description:
*   all_input: initial data (element composition)
*   all_data0: initial label (formation energy)
*   all_data : initial label (ahc)
*   all_data2: initial label (aha)

Output description:

*   top_all    : newly sampled data (element composition)
*   top_select2: final selected sampled data (element composition)
"""

################################ Optimization using DANTE ###############################


def pareto_frontier(data):
    """
    Find the Pareto frontier from a two-dimensional array.

    :param data: A two-dimensional numpy array where rows are points.
    :return: A numpy array with the points on the Pareto frontier.
    """
    # Sort data by the first dimension (x)
    indices = np.argsort(data[:, 0])
    indices = indices[::-1]
    data_sorted = data[data[:, 0].argsort()]
    data_sorted = data_sorted[::-1]
    pareto_front = [data_sorted[0]]
    pareto_indices = [indices[0]]
    for i, point in enumerate(data_sorted[1:]):
        if point[1] > pareto_front[-1][1]:  # Compare with the last point in the Pareto front
            pareto_front.append(point)
            pareto_indices.append(indices[i + 1])
    return np.array(pareto_indices)

def pareto_evaluation(all_input, sample_input,num,plot_save = False): #Euclidean distance + pred score #pareto front
    sample_dist=[]##nearest neighbor distance
    for i in sample_input:
        dist_temp=1000000000000
        for n in all_input:
            dist= np.linalg.norm(i - n)
            if dist < dist_temp:
                dist_temp = round(dist,10)
        sample_dist.append(dist_temp)
    sample_dist=np.array(sample_dist)
    sample_score=oracle(sample_input)
    data=np.concatenate((sample_dist.reshape(-1,1),sample_score.reshape(-1,1)),axis=1)
    print(data.shape)
    pareto_front = pareto_frontier(data)
    while len(pareto_front) < num:
        remaining_data = np.delete(data, pareto_front, axis=0)
        remaining_indices = np.delete(np.arange(data.shape[0]), pareto_front)
        pareto_front2 = pareto_frontier(remaining_data)
        pareto_front = np.concatenate((pareto_front,remaining_indices[pareto_front2]))

    ind = np.random.choice(pareto_front,num,replace=False)

    if plot_save == True:
        plt.figure()
        plt.scatter(sample_dist,sample_score,label='all samples')
        plt.scatter(sample_dist[pareto_front],sample_score[pareto_front],label='pareto-front samples')
        plt.scatter(sample_dist[ind],sample_score[ind],label='selected samples')
        plt.title('distance VS score')
        plt.xlabel('distance')
        plt.ylabel('score')
        plt.legend()
        plt.show()
        plt.savefig(model_folder+'/'+round_name+'/distance VS score.png')
        ##############
        total = np.concatenate((all_input.reshape(-1,n_dim), sample_input.reshape(-1,n_dim)), axis=0)
        print('TSNE shape',total.shape)
        from sklearn.manifold import TSNE
        keep_dims = 2
        prp = 40
        tsne = TSNE(n_components=keep_dims,
                    perplexity=prp,
                    random_state=42,
                    n_iter=5000,
                    n_jobs=-1)
        total_tsne = tsne.fit_transform(total)
        total_tsne0 = total_tsne[:, 0]
        total_tsne1 = total_tsne[:, 1]
        print('TSNE done!')
        plt.figure()
        plt.scatter(total_tsne[:len(all_input), 0], total_tsne[:len(
            all_input), 1], color='black', label='initial data')
        plt.scatter(total_tsne[len(all_input):, 0], total_tsne[len(
            all_input):, 1], color='blue', label='sampled data')
        plt.scatter(total_tsne0[len(all_input)+ind], total_tsne1[len(
            all_input)+ind], color='green', label='top-sampled data')
        plt.title('TSNE ')
        plt.colorbar()
        plt.legend()
        plt.savefig(model_folder+'/'+round_name+'/TSNE.png')

        #################################################################
        from sklearn.decomposition import PCA
        "PCA"
        pca = PCA(n_components=2, whiten=True)
        total_pca = pca.fit_transform(total)
        plt.figure()
        plt.scatter(total_pca[:len(all_input), 0], total_pca[:len(
            all_input), 1], color='black', label='initial data')
        plt.scatter(total_pca[len(all_input):, 0], total_pca[len(
            all_input):, 1], color='blue', label='sampled data')
        plt.scatter(total_pca[len(all_input)+ind, 0], total_pca[len(all_input) +
                    ind, 1], color='green', label='top-sampled data')
        plt.title('PCA')
        plt.colorbar()
        plt.legend()
        plt.savefig(model_folder+'/'+round_name+'/PCA.png')

    return ind


def single_run(X,y,initial_X,greedy_UCT,UCT_low):
    initial_X=initial_X.reshape(n_dim)
    if greedy_UCT== True:
        values = max(y)
        exp_weight = weight * values
    if UCT_low ==True:
        values = oracle(initial_X)
        exp_weight = weight * values

    board_uct = opt_task(tup=tuple(initial_X), value=tuple(oracle(initial_X)), terminal=False)
    tree_ubt = DANTE(exploration_weight=exp_weight)
    boards = []
    boards_rand = np.empty((0,n_dim))
    for i in tqdm(range(0, rollout_round, 1)):
        tree_ubt.do_rollout(board_uct)
        board_uct,board_rand = tree_ubt.choose(board_uct)
        boards.append(list(board_uct.tup))
        boards_rand=np.concatenate((boards_rand,board_rand),axis=0)

    boards_rand = np.unique(boards_rand,axis=0)

    new_x = []
    new_pred = []
    boards = np.array(boards)
    boards = np.unique(boards, axis=0)
    pred_values = oracle(boards)
    print(f'unique number of boards: {len(boards)}')

    for i,j in zip(boards,pred_values):
      temp_x = np.array(i)
      same = np.all(temp_x==all_input.reshape(len(all_input),n_dim), axis=1)
      has_true = any(same)
      if has_true == False:
        new_pred.append(j)
        new_x.append(temp_x)
    new_x= np.array(new_x)
    new_pred = np.array(new_pred)

    top_n=list1[3]
    ind = np.argpartition(new_pred, -top_n)[-top_n:]
    top_prediction =  new_x[ind]

    sample_score=oracle(boards_rand)
    whe=np.where(sample_score > max(sample_score) * 0.8)[0]
    boards_rand2=boards_rand[whe]
    print(len(boards_rand2))
    ind2=pareto_evaluation(all_input, boards_rand2,list1[4])
    top_random2 = boards_rand2[ind2]

    X_most_visit =  most_visit_node(tree_ubt, X.reshape(len(X),n_dim),list1[5])
    X_rand =  random_node(boards_rand,list1[6])
    X_next = np.concatenate([top_prediction,top_random2, X_most_visit, X_rand])
    return X_next,exp_weight

def score_evaluate(ef,ahc,aha):
    ef = np.where(ef>0.15,0,1)
    return ef*ahc*aha

def run(X,ef, ahc,aha, rollout_round):
    score=score_evaluate(ef,ahc,aha)
    greedy_UCT = True
    top_select = list1[1] #highest
    random_select = list1[2] #random
    ind = np.argpartition(score, -top_select)[-top_select:]#####
    ind_random=np.setdiff1d(np.arange(len(score)), ind)
    ind2 = np.random.choice(ind_random,random_select)
    print(ind)
    ind = np.concatenate((ind,ind2))
    print(ind)
    print('ef:',ef[ind])
    print('ahc:',ahc[ind])
    print('aha:',aha[ind])
    x_current_top = X[ind]
    y_top=score[ind]
    X_top=[]
    top_selections = []
    for i in range(len(ind)):
      top_temp = x_current_top[i]
      print("true of top:",y_top[i])
      print("top_temp:",top_temp)
      x,exp_weight = single_run(X,score,top_temp,greedy_UCT,UCT_low)
      print('select alloy contents:',x)
      print(oracle(x))
      X_top.append(x)
      top_selections.append(top_temp)

    top_X = np.vstack(X_top)
    print(top_X.shape)
    print(f'exp_weight is {exp_weight}')
    print(f'top x are {top_selections}')
    print(f'top selection are {X_top}')
    return top_X


ele_range = component_generate()
top_all=[]
for i in range(list1[0]):
    top_X=run(all_input,all_data0.reshape(-1),all_data1.reshape(-1),all_data2.reshape(-1),rollout_round)
    print(top_X)
    top_all.append(top_X)

################################ Select final samples ###############################

top_all2=np.array(top_all).reshape(-1,n_dim)
top_all3=np.unique(top_all2,axis=0)
sample_score=oracle(top_all3)

whe=np.where(sample_score > max(sample_score) * 0.5)[0]
top_all4=top_all3[whe]
ind=pareto_evaluation(all_input, top_all4,top_sample,plot_save=True)
top_select = top_all4[ind]
print('score:',oracle(top_select))

top_select2=top_select.reshape(-1,n_dim)
np.save(model_folder+'/'+round_name+'/top_select.npy', top_select2, allow_pickle=True)

df = pd.DataFrame(top_select2)
df.columns= ['Ti','Nb','Al','Ge','Co','Au','Pd','Ni','Zn','Ga','Mo','Cu','Pt','Sn','Cr','Mn','Mg','Si','Fe','Ru','Rh','Hf','Ta','W','Re','Ir','Bi']
df.to_csv(model_folder+'/'+round_name+'/top_select_fcc.csv')

################################ Visualization ###############################

sample_ef=emsemble_pred0(top_all3) # predicted formation energy of sampled points
sample_ahc=emsemble_pred1(top_all3) # predicted AHC of sampled points
sample_aha=emsemble_pred2(top_all3) # predicted AHA of sampled points
ef1 = np.where(sample_ef>0.2,0,1)


top_sample_ef=emsemble_pred0(top_select2) # predicted formation energy of top-sampled points
top_sample_ahc=emsemble_pred1(top_select2) # predicted AHC of top-sampled points
top_sample_aha=emsemble_pred2(top_select2) # predicted AHA of top-sampled points
ef2 = np.where(top_sample_ef>0.2,0,1)


initial_ef=emsemble_pred0(all_input) # predicted formation energy of initial points
initial_ahc=emsemble_pred1(all_input) # predicted AHC of initial points
initial_aha=emsemble_pred2(all_input) # predicted AHA of initial points
ef0 = np.where(initial_ef>0.2,0,1)



plt.figure()
plt.scatter(initial_ahc / 10000, ef0 * initial_aha / 100, label='initial data')
plt.scatter(sample_ahc / 10000, ef1 * sample_aha / 100, label='sampled data')
plt.scatter(top_sample_ahc / 10000,  ef2 * top_sample_aha / 100, label='top-sampled data')
plt.xlabel('predicted AHC')
plt.ylabel('predicted AHA (ef < 0.02)')
plt.legend()

################################ End of Part IV ################################

"""################################################################################

---------------------------------------------------------------------------- That's all folks ! ----------------------------------------------------------------------------


################################################################################
"""


