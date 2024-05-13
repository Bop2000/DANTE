"""################################################################################
> # **Introduction**
> The notebook is divided into 4 major parts :

*   **Part I** : voxelization of the architected materials
*   **Part II** : define DOTS algorithm
*   **Part III** : define and train the CNN model
*   **Part IV** : optimization using DOTS

################################################################################

################################################################################
> # **Part - I**

*   Import initial dataset
*   Set parameters
*   Voxelization of the architected materials

################################################################################
"""

############################### Import libraries ###############################

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense,Dropout,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler,ModelCheckpoint
import seaborn as sns
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

model_folder = "Results"
# Check if the directory exists
if not os.path.exists(model_folder):
    # If it doesn't exist, create it
    os.makedirs(model_folder)
if not os.path.exists(model_folder+'/'+round_name):
    # If it doesn't exist, create it
    os.makedirs(model_folder+'/'+round_name)
    







############################### Import Initial Dataset ###############################

all_input=np.load("input333.npy", allow_pickle=True)
data = pd.read_excel("data.xlsx")

rate=100
all_data = np.array(data['E'])/rate # rescaled for cnn model training
all_data2 = np.array(data['Y'])

if round_num  == 1:
  plt.figure()
  plt.scatter(all_data*rate,all_data2,label='Inital data')
  plt.legend()
  plt.title('E vesus Y')
  plt.xlabel('Elastic Modulus (MPa)')
  plt.ylabel('Yield Strength (MPa)')
  plt.savefig(f'round{round_num-1}.png')
else:
  plt.figure()
  plt.scatter(all_data*rate,all_data2,label='before')
  plt.scatter(all_data[-20:],all_data2[-20:],label=f'round{round_num-1}')
  plt.legend()
  plt.title('E vesus Y')
  plt.xlabel('Elastic Modulus (MPa)')
  plt.ylabel('Yield Strength (MPa)')
  plt.savefig(f'round{round_num-1}.png')







############################### Set Paramaters ###############################

top_sample=20 # number of newly selected samples
n_model=5 # number of cnn models for predictions
n_dim=27  # Dimension of this optimization problem
rollout_round=100 #number of roullout steps for DOTS algorithm,By default, DOTS performs 100 rollout
UCT_low=False
weight = 0.2 # exploration weight = weight * max(score)
list1=[5,8,2,5,1,1] ##[run times, top start points, random start points, top score samples, top visit samples, random samples]
target = 2500 #Choose a elastic modulus target, such as target = 2500 MPa
n_size=6 # dimension of the architected materials to optimize, where is 6 mm here
n_accu=60 # The number of points in the three directions of x, y and z after voxelization of the architected material
xxx=(1/2)*2*pi
sizeofdata0=[3,3,3] # an architected material with 3*3*3 units
accu=int(n_accu / 3)
x_axis, y_axis,z_axis = np.linspace(n_size/(n_accu*2), n_size-n_size/(n_accu*2), n_accu),  np.linspace(n_size/(n_accu*2), n_size-n_size/(n_accu*2), n_accu),  np.linspace(n_size/(n_accu*2), n_size-n_size/(n_accu*2), n_accu)
x, y,z = np.meshgrid(x_axis, y_axis,z_axis)






############################### voxelization of the architected materials ###############################

def findneighbour(inputdata,position):
    neighbourhoods=np.zeros((3,3,3))
    neighbourhoods[:,:,:]=np.nan
    r=len(inputdata)
    flag=0
    for i in range(r):
        if inputdata[i,0]==position[0] and inputdata[i,1]==position[1] and inputdata[i,2]==position[2]:
            flag=1
    if flag!=0:
        for i in range(r):
            dertax=inputdata[i,0]-position[0]
            dertay=inputdata[i,1]-position[1]
            dertaz=inputdata[i,2]-position[2]
            if abs(dertax)<=1 and abs(dertay)<=1 and abs(dertaz)<=1:
                neighbourhoods[int(dertax+1),int(dertay+1),int(dertaz+1)]=inputdata[i,3]
    return neighbourhoods

def createunitofv(datainput,positon,nofv,dofv):
    neibourhoods=findneighbour(datainput,positon)
    unitofv=np.ones((nofv-2*dofv,nofv-2*dofv,nofv-2*dofv))
    if not np.isnan(neibourhoods[1,1,1]):
        unitofv=unitofv*neibourhoods[1,1,1]
    else:
        unitofv=np.zeros((nofv,nofv,nofv))
        unitofv[:,:,:]=np.nan
        return unitofv
    if np.isnan(neibourhoods[2,1,1]):
        neibourhoods[2,1,1]=neibourhoods[1,1,1]
    if np.isnan(neibourhoods[0,1,1]):
        neibourhoods[0,1,1]=neibourhoods[1,1,1]
    if np.isnan(neibourhoods[1,2,1]):
        neibourhoods[1,2,1]=neibourhoods[1,1,1]
    if np.isnan(neibourhoods[1,0,1]):
        neibourhoods[1,0,1]=neibourhoods[1,1,1]
    if np.isnan(neibourhoods[1,1,2]):
        neibourhoods[1,1,2]=neibourhoods[1,1,1]
    if np.isnan(neibourhoods[1,1,0]):
        neibourhoods[1,1,0]=neibourhoods[1,1,1]
    if np.isnan(neibourhoods[2,2,1]):
        neibourhoods[2,2,1]=(neibourhoods[2,1,1]+neibourhoods[1,2,1])/2
    if np.isnan(neibourhoods[2,0,1]):
        neibourhoods[2,0,1]=(neibourhoods[2,1,1]+neibourhoods[1,0,1])/2
    if np.isnan(neibourhoods[0,2,1]):
        neibourhoods[0,2,1]=(neibourhoods[0,1,1]+neibourhoods[1,2,1])/2
    if np.isnan(neibourhoods[0,0,1]):
        neibourhoods[0,0,1]=(neibourhoods[0,1,1]+neibourhoods[1,0,1])/2
    if np.isnan(neibourhoods[2,1,2]):
        neibourhoods[2,1,2]=(neibourhoods[2,1,1]+neibourhoods[1,1,2])/2
    if np.isnan(neibourhoods[2,1,0]):
        neibourhoods[2,1,0]=(neibourhoods[2,1,1]+neibourhoods[1,1,0])/2
    if np.isnan(neibourhoods[0,1,2]):
        neibourhoods[0,1,2]=(neibourhoods[0,1,1]+neibourhoods[1,1,2])/2
    if np.isnan(neibourhoods[0,1,0]):
        neibourhoods[0,1,0]=(neibourhoods[0,1,1]+neibourhoods[1,1,0])/2
    if np.isnan(neibourhoods[1,2,2]):
        neibourhoods[1,2,2]=(neibourhoods[1,2,1]+neibourhoods[1,1,2])/2
    if np.isnan(neibourhoods[1,2,0]):
        neibourhoods[1,2,0]=(neibourhoods[1,2,1]+neibourhoods[1,1,0])/2
    if np.isnan(neibourhoods[1,0,2]):
        neibourhoods[1,0,2]=(neibourhoods[1,0,1]+neibourhoods[1,1,2])/2
    if np.isnan(neibourhoods[1,0,0]):
        neibourhoods[1,0,0]=(neibourhoods[1,0,1]+neibourhoods[1,1,0])/2
    if np.isnan(neibourhoods[0,0,0]):
        neibourhoods[0,0,0]=(neibourhoods[0,1,1]+neibourhoods[1,0,1]+neibourhoods[1,1,0])/3
    if np.isnan(neibourhoods[2,0,0]):
        neibourhoods[2,0,0]=(neibourhoods[2,1,1]+neibourhoods[1,0,1]+neibourhoods[1,1,0])/3
    if np.isnan(neibourhoods[0,2,0]):
        neibourhoods[0,2,0]=(neibourhoods[0,1,1]+neibourhoods[1,2,1]+neibourhoods[1,1,0])/3
    if np.isnan(neibourhoods[0,0,2]):
        neibourhoods[0,0,2]=(neibourhoods[0,1,1]+neibourhoods[1,0,1]+neibourhoods[1,1,2])/3
    if np.isnan(neibourhoods[0,2,2]):
        neibourhoods[0,2,2]=(neibourhoods[0,1,1]+neibourhoods[1,2,1]+neibourhoods[1,1,2])/3
    if np.isnan(neibourhoods[2,0,2]):
        neibourhoods[2,0,2]=(neibourhoods[2,1,1]+neibourhoods[1,0,1]+neibourhoods[1,1,2])/3
    if np.isnan(neibourhoods[2,2,0]):
        neibourhoods[2,2,0]=(neibourhoods[2,1,1]+neibourhoods[1,2,1]+neibourhoods[1,1,0])/3
    if np.isnan(neibourhoods[2,2,2]):
        neibourhoods[2,2,2]=(neibourhoods[2,1,1]+neibourhoods[1,2,1]+neibourhoods[1,1,2])/3
    for i in range(dofv):
        nownumber=neibourhoods[1,1,1]+i*(neibourhoods-neibourhoods[1,1,1])/(2*dofv+1)
        temp=np.zeros((1,nofv-2*dofv+2*i,nofv-2*dofv+2*i))
        temp[:,:,:]=nownumber[2,1,1]
        unitofv=np.concatenate((unitofv,temp),axis=0)#x+
        temp[:,:,:]=nownumber[0,1,1]
        unitofv=np.concatenate((temp,unitofv),axis=0)#x-
        temp=np.zeros((nofv-2*dofv+2*i+2,1,nofv-2*dofv+2*i))
        temp[:,:,:]=nownumber[1,2,1]
        unitofv=np.concatenate((unitofv,temp),axis=1)#y+
        temp[:,:,:]=nownumber[1,0,1]
        unitofv=np.concatenate((temp,unitofv),axis=1)#y-
        temp=np.zeros((nofv-2*dofv+2*i+2,nofv-2*dofv+2*i+2,1))
        temp[:,:,:]=nownumber[1,1,2]
        unitofv=np.concatenate((unitofv,temp),axis=2)#z+
        temp[:,:,:]=nownumber[1,1,0]
        unitofv=np.concatenate((temp,unitofv),axis=2)#z-
        unitofv[[-1],[-1],:]=nownumber[2,2,1]#x+,y+
        unitofv[0,0,:]=nownumber[0,0,1]#x-,y-
        unitofv[[-1],0,:]=nownumber[2,0,1]#x+,y-
        unitofv[0,[-1],:]=nownumber[0,2,1]#x,y+
        unitofv[[-1],:,[-1]]=nownumber[2,1,2]
        unitofv[0,:,0]=nownumber[0,1,0]
        unitofv[[-1],:,0]=nownumber[2,1,0]
        unitofv[0,:,[-1]]=nownumber[0,1,2]
        unitofv[:,[-1],[-1]]=nownumber[1,2,2]
        unitofv[:,0,0]=nownumber[1,0,0]
        unitofv[:,[-1],0]=nownumber[1,2,0]
        unitofv[:,0,[-1]]=nownumber[1,0,2]
        unitofv[[-1],[-1],[-1]]=nownumber[2,2,2]
        unitofv[0,[-1],[-1]]=nownumber[0,2,2]
        unitofv[[-1],0,[-1]]=nownumber[2,0,2]
        unitofv[[-1],[-1],0]=nownumber[2,2,0]
        unitofv[[-1],0,0]=nownumber[2,0,0]
        unitofv[0,[-1],0]=nownumber[0,2,0]
        unitofv[0,0,[-1]]=nownumber[0,0,2]
        unitofv[0,0,0]=nownumber[0,0,0]
    return unitofv

def createv_2(data,sizeofdata,nofv,dofv):
    v=[]
    for k in range(sizeofdata[2]):
        temp2=[]
        for j in range(sizeofdata[1]):
            temp1=[]
            for i in range(sizeofdata[0]):
                position=[i,j,k]
                varray=createunitofv(data,position,nofv,dofv)
                if i<1:
                    temp1=varray
                else:
                    temp1=np.concatenate((temp1,varray),axis=0)
            if j<1:
                temp2=temp1
            else:
                temp2=np.concatenate((temp2,temp1),axis=1)
        if k<1:
            v=temp2
        else:
            v=np.concatenate((v,temp2),axis=2)
    return v

############
r1=np.zeros((27,3))
for a in range(3):
    for b in range(3):
        for c in range(3):
            r1[9*a+3*b+c,0]=a
            r1[9*a+3*b+c,1]=b
            r1[9*a+3*b+c,2]=c
#############
oo=np.sin(pi*x)*np.cos(pi*y)+np.sin(pi*y)*np.cos(pi*z)+np.sin(pi*z)*np.cos(pi*x)
def To60(matrix):
    the606060=[]
    N=len(matrix)
    # r1_100=np.tile(r1, (N,1,1))
    finished=(10*(1-matrix).reshape(N,27,1))*0.282-0.469
    # print(finished.shape)
    # data_all=np.concatenate((r1_100,finished),axis=2)
    for l in range(N):
        r2=finished[l]
        data0=np.concatenate((r1,r2),axis=1)
        v=createv_2(data0,sizeofdata0,accu,3)
        ov=oo+v
        the606060.append(ov)
    the606060_cell=np.asarray(the606060)
    the606060_cell=np.where(the606060_cell<0.9,1,0)
    return the606060_cell

################################# End of Part I ################################








"""################################################################################
> # **Part - II**

*   Define the DOTS alghorithm

################################################################################
"""

################################# DOTS alghorithm ################################


class DOTS:
    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

    def choose(self, node):
        "Choose the best successor of node."
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            print('not seen before, randomly sampled!')
            return node.find_random_child()

        def evaluate(n):
            return n.value  # average reward
        print(f'number of visit is {self.N[node]}')
        log_N_vertex = math.log(self.N[node])
        def uct(n):
            "Upper confidence bound for trees"

            uct_value = n.value + self.exploration_weight * math.sqrt(
                log_N_vertex / (self.N[n]+1))
            return uct_value

        action = [p for p in range(0, len(node.tup))]
        self.children[node] = node.find_children(action)
        media_node = max(self.children[node], key=uct)#self._uct_select(node)
        rand_index = random.randint(0, len(list(self.children[node]))-1)
        node_rand = list(self.children[node])[rand_index]

        if uct(media_node) > uct(node):
            print(f'media node is{media_node}')
            print(f'uct of the node is{uct(media_node)} ')
            print(f'better value media node : {media_node.value}')
            return media_node, node_rand
        return node, node_rand


    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        count = 0
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
              return path
            unexplored = self.children[node] - self.children.keys()
            def evaluate(n):
              return n.value
            if count == 50:
             return max(path, key=evaluate)

            if unexplored:
              path.append(max(unexplored, key=evaluate))#
              return path
            node = self._uct_select(node)  # descend a layer deeper
            count+=1

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        action = [p for p in range(0, len(node.tup))]
        self.children[node] = node.find_children(action)

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        reward = node.reward()
        return reward

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
          self.N[node] += 1
          self.Q[node] += reward
    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"
        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])
        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            uct_value = n.value + self.exploration_weight * math.sqrt(
                log_N_vertex / (self.N[n]+1))
            return uct_value
        uct_node = max(self.children[node], key=uct)
        print(f'node with max uct is:{uct_node}')
        return uct_node

class Node(ABC):
    """
    A representation of a single board state.
    DOTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True

_OT = namedtuple("opt_task", "tup value terminal")
class opt_task(_OT, Node):

    ############################ design the action space ############################
    # for this task, xi belongs to [0.1,0.8] with an interval of 0.1

    def find_children(board,action):
        if board.terminal:
            return set()
        turn = 0.1
        all_tup=[]
        for index in action:
            tup = list(board.tup)
            flip = random.randint(0,6)
            if   flip ==0:
              tup[index] += turn

            elif flip ==1:
                tup[index] -= turn

            elif flip ==2:
              for i in range(int(n_dim/5)):
                index_2 = random.randint(0, len(tup)-1)
                tup[index_2] = random.randint(1, 8)/10
            elif flip ==3:
              for i in range(int(n_dim/10)):
                index_2 = random.randint(0, len(tup)-1)
                tup[index_2] = random.randint(1, 8)/10
            elif flip ==5:
              tup[index] = random.randint(1, 8)/10
            else:
              tup[index] = random.randint(1, 8)/10
            tup[index] = round(tup[index],2)
            tup=np.array(tup)
            whe1=np.where(tup<0.1)
            whe2=np.where(tup>0.8)
            tup[whe1[0]]=0.1
            tup[whe2[0]]=0.8
            all_tup.append(tup)

        all_value = oracle(all_tup)
        is_terminal=False
        return  {opt_task(tuple(t), v, is_terminal) for t, v in  zip(all_tup,all_value)}

    def reward(board):
        return  oracle(board.tup)
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

def model_training(X,y,name,i,lr):
      ind=index_random[round(i*len(index_random)/5):round((1+i)*len(index_random)/5)]####1/5 data as test set
      ind2=np.setdiff1d(index_random, ind)
      X_train, X_test, y_train, y_test = X[ind2],X[ind], y[ind2],y[ind]
      inputs = keras.Input((60, 60, 60, 1))
      x = layers.Conv3D(filters=8, kernel_size=3, activation="elu",padding='same')(inputs)
      x = layers.MaxPool3D(pool_size=2,padding='same')(x)
      x = layers.Conv3D(filters=4, kernel_size=3, activation="elu",padding='same')(x)
      x = layers.MaxPool3D(pool_size=2,padding='same')(x)
      x = layers.Conv3D(filters=2, kernel_size=3, activation="elu",padding='same')(x)
      x = layers.MaxPool3D(pool_size=2,padding='same')(x)
      x = layers.Flatten()(x)
      x = layers.Dense(units=128, activation="elu")(x)
      x = layers.Dense(units=64, activation="elu")(x)
      x = layers.Dense(units=32, activation="elu")(x)
      outputs = layers.Dense(units=1, activation="linear")(x)
      model = keras.Model(inputs, outputs, name="3dcnn")
      mc = ModelCheckpoint(model_folder+'/'+round_name+f"/{name}.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)
      early_stop = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
      model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
      model.fit(X_train, y_train, batch_size=32, epochs=5000, validation_data=(X_test, y_test), callbacks=[early_stop,mc])
      model=keras.models.load_model(model_folder+'/'+round_name+f"/{name}.h5")

      R2,MAE=mar_r2(model,X_test,y_test)

      return model,X_test,y_test,R2,MAE

def model_performance(model,X_test,y_test):
    perform_list=pd.read_csv(model_folder+'/'+round_name+f'/model_performance_{n_dim}d.csv')
    y_pred = model.predict(X_test.reshape(len(X_test),60,60,60,1))
    R2=stats.pearsonr(y_pred.reshape(-1), y_test.reshape(-1))[0]
    R2=np.asarray(R2).round(5)
    MAE= metrics.mean_absolute_error(y_test.reshape(-1), y_pred.reshape(-1))

    plt.figure()
    sns.set()
    sns.regplot(x=y_pred, y=y_test, color='k')
    plt.title(('R2:',R2,'MAE:',MAE))

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


def mar_r2(model,X_test,y_test):
    y_pred = model.predict(X_test.reshape(len(X_test),60,60,60,1))
    R2=stats.pearsonr(y_pred.reshape(-1), y_test.reshape(-1))[0]
    R2=np.asarray(R2).round(5)
    MAE= metrics.mean_absolute_error(y_test.reshape(-1), y_pred.reshape(-1))
    return R2,MAE







################################ Train the CNN model ################################

X60 = To60(np.array(all_input).reshape(-1,3,3,3)) # voxelization

### train 5 models for Elastic modulus prediction
pd.DataFrame(np.empty(0)).to_csv(model_folder+'/'+round_name+f'/model_performance_{n_dim}d.csv')
for i in range(5):
    trytime=0
    model1,X_test1,y_test1,R21,MAE1 = model_training(X60,all_data,f'E{i}',i,0.001) # train the model of E
    R20=R21
    while R21 < 0.95 and trytime<5:
        trytime+=1
        model1,X_test1,y_test1,R21,MAE1 = model_training(X60,all_data,f'E{i+10}',i,0.002) # train the model of E
        if R21>R20:
            R20=R21
            model1.save("Results/"+round_name+f'/E{i}.h5')
    model1=keras.models.load_model(model_folder+'/'+round_name+f'/E{i}.h5')
    R21,MAE1=model_performance(model1,X_test1,y_test1) # show and save the performance of the model

### train 5 models for Yield strength prediction
for i in range(5):
    trytime=0
    model2,X_test2,y_test2,R22,MAE2 = model_training(X60,all_data2,f'Y{i}',i,0.005) # train the model of Y
    R20=R22
    while R22 < 0.95 and trytime<5:
        trytime+=1
        model2,X_test2,y_test2,R22,MAE2 = model_training(X60,all_data2,f'Y{i+10}',i,0.002) # train the model of Y
        if R22>R20:
            R20=R22
            model2.save("Results/"+round_name+f'/Y{i}.h5')
    model2=keras.models.load_model(model_folder+'/'+round_name+f'/Y{i}.h5')
    R22,MAE2=model_performance(model2,X_test2,y_test2) # show and save the performance of the model






################################ Load and use the CNN model ################################

path = os.getcwd()
name2=path+'/Results/'+round_name+'/'
models=dict()
model_E_list=[]
for i in range(0,n_model):
    modelname = f'E{i}'
    model_E_list.append(modelname)
    models[modelname]= keras.models.load_model(name2+modelname+'.h5')
model_Y_list=[]
for i in range(0,n_model):
    modelname = f'Y{i}'
    model_Y_list.append(modelname)
    models[modelname]= keras.models.load_model(name2+modelname+'.h5')

###emsemble all models to predict
def emsemble_predict1(S,n_model=5):##E
    pred_all=0
    for i in range(n_model):
        temp=models[model_E_list[i]].predict(S.reshape(len(S),60,60,60,1))
        pred_all+=temp
    pred_all/=n_model
    return pred_all

def emsemble_predict2(S,n_model=5):##Y
    pred_all=0
    for i in range(n_model):
        temp=models[model_Y_list[i]].predict(S.reshape(len(S),60,60,60,1))
        pred_all+=temp
    pred_all/=n_model
    return pred_all

def oracle(x):
    try:
       x60=To60(np.array(x).reshape(len(x),3,3,3))
       pred = emsemble_predict1(np.array(x60).reshape(len(x60),60,60,60,1))
       pred = np.array(pred).reshape(len(x60))
       whe=np.where((pred>2650/rate)|(pred<2350/rate))
       pred2 = emsemble_predict2(np.array(x60).reshape(len(x60),60,60,60,1))
       pred2 = np.array(pred2).reshape(len(x60))
       pred2[whe[0]]=0
    except:
       x60=To60(np.array(x).reshape(1,3,3,3))
       pred = emsemble_predict1(np.array(x60).reshape(1,60,60,60,1))
       pred = np.array(pred).reshape(1)
       pred2 = emsemble_predict2(np.array(x60).reshape(1,60,60,60,1))
       pred2 = np.array(pred2).reshape(1)
       if pred<2350/rate or pred>2650/rate:
            pred2=0
    print('pred E:',pred)
    print('pred Y:',pred2)
    return pred2






################################ model performance visualization ################################

def model_visual1(X,y,i): # E
    model1=models[model_E_list[i]]
    ind=index_random[round(i*len(index_random)/5):round((1+i)*len(index_random)/5)]####1/5 data as test set
    ind2=np.setdiff1d(index_random, ind)
    X_train, X_test, y_train, y_test = X[ind2],X[ind], y[ind2],y[ind]

    y_pred = model1.predict(X_test.reshape(len(X_test),60,60,60,1))
    R2=stats.pearsonr(y_pred.reshape(-1), y_test.reshape(-1))[0]
    R2=np.asarray(R2).round(3)
    MAE= metrics.mean_absolute_error(y_test.reshape(-1), y_pred.reshape(-1))
    MAE=np.asarray(MAE).round(5)
    MAPE= metrics.mean_absolute_percentage_error(y_test.reshape(-1), y_pred.reshape(-1))
    MAPE=np.asarray(MAPE).round(5)
    plt.figure()
    sns.set()
    sns.regplot(x=y_pred, y=y_test, color='k')
    plt.title((f'E prediction by model #{i+1}: ','R2:',R2,'MAE:',MAE,'MAPE:',MAPE))
    plt.xlabel('Predicted elasctic modulus')
    plt.xlabel('Simulated elasctic modulus')

def model_visual2(X,y,i):# Y
    model2=models[model_Y_list[i]]
    ind=index_random[round(i*len(index_random)/5):round((1+i)*len(index_random)/5)]####1/5 data as test set
    ind2=np.setdiff1d(index_random, ind)
    X_train, X_test, y_train, y_test = X[ind2],X[ind], y[ind2],y[ind]

    y_pred = model2.predict(X_test.reshape(len(X_test),60,60,60,1))
    R2=stats.pearsonr(y_pred.reshape(-1), y_test.reshape(-1))[0]
    R2=np.asarray(R2).round(3)
    MAE= metrics.mean_absolute_error(y_test.reshape(-1), y_pred.reshape(-1))
    MAE=np.asarray(MAE).round(5)
    MAPE= metrics.mean_absolute_percentage_error(y_test.reshape(-1), y_pred.reshape(-1))
    MAPE=np.asarray(MAPE).round(5)
    plt.figure()
    sns.set()
    sns.regplot(x=y_pred, y=y_test, color='k')
    plt.title((f'Y prediction by model #{i+1}: ','R2:',R2,'MAE:',MAE,'MAPE:',MAPE))
    plt.xlabel('Predicted yield strength')
    plt.xlabel('Simulated yield strength')

models[model_Y_list[0]].summary()
for i in range(5):
    model_visual1(X60,all_data,i)
for i in range(5):
    model_visual2(X60,all_data2,i)

################################ End of Part III ###############################








"""################################################################################
> # **Part - IV**

*   Optimization using DOTS

################################################################################

Input description:
*   all_input: initial data (Density matrix)
*   X60      : voxelized architected materials
*   all_data : initial label (Elastic modulus)
*   all_data2: initial label (Yield strength)

Output description:

*   top_all    : newly sampled data (Density matrix)
*   top_select2: final selected sampled data (Density matrix)
"""





################################ Optimization using DOTS ###############################


def single_run(X,y,initial_X,initial_y,greedy_UCT,UCT_low):
    initial_X=initial_X.reshape(27)
    if greedy_UCT== True:
        values = max(y)
        exp_weight = weight * values
    if UCT_low ==True:
        values = oracle(initial_X)
        exp_weight = weight * values

    board_uct = opt_task(tup=tuple(initial_X), value=initial_y, terminal=False)
    tree_ubt = DOTS(exploration_weight=exp_weight)
    boards = []
    boards_rand = []
    for i in tqdm(range(0, rollout_round, 1)):
        tree_ubt.do_rollout(board_uct)
        board_uct,board_rand = tree_ubt.choose(board_uct)
        boards.append(list(board_uct.tup))
        boards_rand.append(list(board_rand.tup))

    new_x = []
    new_pred = []
    boards = np.array(boards)
    boards = np.unique(boards, axis=0)
    pred_values = oracle(boards)
    print(f'unique number of boards: {len(boards)}')

    for i,j in zip(boards,pred_values):
      temp_x = np.array(i)
      same = np.all(temp_x==X.reshape(len(X),27), axis=1)
      has_true = any(same)
      if has_true == False:
        new_pred.append(j)
        new_x.append(temp_x)
    new_x= np.array(new_x)
    new_pred = np.array(new_pred)

    top_n=list1[3]
    ind = np.argpartition(new_pred, -top_n)[-top_n:]
    top_prediction =  new_x[ind]
    X_most_visit =  most_visit_node(tree_ubt, X.reshape(len(X),27),list1[4])
    X_rand =  random_node(new_x,list1[5])
    X_next = np.concatenate([top_prediction, X_most_visit, X_rand])
    return X_next,exp_weight


def run(X60,X,y,yy2, rollout_round):

    y2=np.array(yy2)
    whe=np.where((y>2650/rate)|(y<2350/rate))
    y2[whe[0]]=0

    greedy_UCT = True

    top_select = list1[1] #highest
    random_select = list1[2] #random
    ind = np.argpartition(y2, -top_select)[-top_select:]#####
    ind_random=np.setdiff1d(np.arange(len(y2)), ind)
    ind2 = np.random.choice(ind_random,random_select)
    print(ind)
    ind = np.concatenate((ind,ind2))
    print(ind)

    x_current_top = X[ind]
    y_top=y2[ind]
    X_top=[]
    top_selections = []
    for i in range (top_select+random_select):
      top_temp = x_current_top[i]
      print("true of top:",y_top[i])
      print("top_temp:",top_temp)
      x,exp_weight = single_run(X,y2,top_temp,y_top[i],greedy_UCT,UCT_low)
      X_top.append(x)
      top_selections.append(top_temp)

    top_X = np.vstack(X_top)
    print(top_X.shape)
    print(f'exp_weight is {exp_weight}')
    print(f'top x are {top_selections}')
    print(f'true value of  top x are {y[ind]}')
    print(f'top selection are {X_top}')
    return top_X



top_all=[]
for i in range(list1[0]):
    top_X=run(X60,all_input,all_data,all_data2,rollout_round)
    print(top_X)
    top_all.append(top_X)





################################ Select final samples ###############################

def TSNEPCA(all_input, sample_input, sample_score):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    #################################################################
    "TSNE"
    total = np.concatenate((all_input.reshape(-1,27), sample_input.reshape(-1,27)), axis=0)
    print(total.shape)
    # set the hyperparmateres
    keep_dims = 2
    lrn_rate = 700
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
    ########################### nearest neighbor distance ranking ###########################
    sample_dist=[]##nearest neighbor distance
    for i in range(len(sample_input)):
        x1 = total_tsne0[len(all_input)+i]
        y1 = total_tsne1[len(all_input)+i]
        # print('x1y1',x1,y1)
        dist_temp=1000000000000
        for n in range(len(all_input)):
            x2 = total_tsne0[n]
            y2 = total_tsne1[n]
            dist= ((x1-x2)**2 + (y1-y2)**2)**0.5
            if dist < dist_temp:
                dist_temp = round(dist,10)
                # print(x2,y2)
        sample_dist.append(dist_temp)
    sample_dist=np.array(sample_dist)
    dist_rank=np.argsort(sample_dist)

    ############################### score ranking ###########################
    score_rank=np.argsort(sample_score.reshape(-1))

    ############################## ranking by score and distance ###########################
    all_rank=dist_rank+score_rank
    top_select = round(top_sample*3/4)
    random_select = round(top_sample/4)
    ind = np.argpartition(all_rank, -top_select)[-top_select:]
    ind_random=np.setdiff1d(np.arange(len(all_rank)), ind)
    ind2 = np.random.choice(ind_random,random_select)
    print(ind)
    ind = np.concatenate((ind,ind2))
    print(ind)

    plt.figure()
    plt.hist(sample_dist, bins=50, color='green', edgecolor='black',
             label='all samples')
    plt.hist(sample_dist[ind], bins=30, color='yellow', edgecolor='black',
             label='top samples')
    plt.title('nearest neighbor distance')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    plt.savefig(model_folder+'/'+round_name+'/nearest neighbor distance.png')
    ##############
    plt.figure()
    plt.scatter(total_tsne[:len(all_input), 0], total_tsne[:len(
        all_input), 1], color='black', label='initial data')
    plt.scatter(total_tsne[len(all_input):, 0], total_tsne[len(
        all_input):, 1], color='blue', label='sampled data')
    plt.scatter(total_tsne0[len(all_input)+ind], total_tsne1[len(
        all_input)+ind], color='green', label='top-sampled data')
    plt.title('TSNE ')
    # plt.colorbar()
    plt.legend()
    plt.savefig(model_folder+'/'+round_name+'/TSNE.png')

    ##############################################################
    #################################################################
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
    # plt.colorbar()
    plt.legend()
    plt.savefig(model_folder+'/'+round_name+'/PCA.png')

    return ind



top_all2=np.array(top_all).reshape(-1,3,3,3)
top_all3=np.unique(top_all2,axis=0)

mat60=To60(top_all3)
sample_score=emsemble_predict2(mat60)

ind=TSNEPCA(all_input, np.array(top_all3).reshape(-1,27), np.array(sample_score).reshape(-1))
print('score:',sample_score[ind])
top_select = top_all3[ind].reshape(-1,27)
top_select2=top_select.reshape(-1,3,3,3)
import scipy.io
scipy.io.savemat(model_folder+'/'+round_name+'/top_matrix_3x3x3.mat', {'mydata':np.array(top_select2)})





################################ Visualization ###############################

sample_score0=emsemble_predict1(mat60) # predicted E values of sampled points


mat60_top=To60(top_select2)
top_sample_E = emsemble_predict1(mat60_top) # predicted E values of top-sampled points
top_sample_Y = emsemble_predict2(mat60_top) # predicted Y values of top-sampled points

initial_E = emsemble_predict1(X60) # predicted E values of initial points
initial_Y = emsemble_predict2(X60) # predicted Y values of initial points

plt.figure()
plt.scatter(initial_E * 100, initial_Y, label='initial data')
plt.scatter(sample_score0 * 100,sample_score, label='sampled data')
plt.scatter(top_sample_E * 100, top_sample_Y, label='top-sampled data')
plt.xlabel('predicted elastic modulus (MPa)')
plt.ylabel('predicted yield strength (MPa)')
plt.legend()

################################ End of Part IV ################################

"""################################################################################

---------------------------------------------------------------------------- That's all folks ! ----------------------------------------------------------------------------


################################################################################
"""