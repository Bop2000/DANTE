import numpy as np
from collections import namedtuple
from abc import ABC, abstractmethod
from collections import defaultdict
import math
import random



#######################################################
'DOTS'
#######################################################
class MCTS_ubt:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=None,f=None, model = None, name = None):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight
        self.model = model
        self.f = f
        self.name = name

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            print('not seen before, randomly sampled!')
            return node.find_random_child()

        def evaluate(n):
            return n.value  # average reward
        log_N_vertex = math.log(self.N[node])
        def uct(n):
            "Upper confidence bound for trees"
            uct_value = n.value + self.exploration_weight * math.sqrt(
                log_N_vertex / (self.N[n]+1))
            return uct_value
        
        ###########################################################
        action = [p for p in range(0, len(node.tup))]
        self.children[node] = node.find_children(action,self.f,self.model)  
        ###########################################################
        
        media_node = max(self.children[node], key=uct)
        node_rand = []
        # for i in range(len(list(self.children[node]))):
        ind=np.random.randint(0,len(list(self.children[node])),2) ##for computer memory consideration
        for i in ind:
              node_rand.append(list(self.children[node])[i].tup)         
        
        if uct(media_node) > uct(node):
            return media_node, node_rand
        return node, node_rand

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)
        
    def data_process(self,X,boards):
        new_x = []
        boards = np.array(boards)
        boards = np.unique(boards, axis=0)
        for i in boards:
          temp_x = np.array(i)
          same = np.all(temp_x==X, axis=1)
          has_true = any(same)
          if has_true == False:
            new_x.append(temp_x)
        new_x= np.array(new_x)
        print(f'unique number of boards: {len(new_x)}')
        return new_x
    
    def most_visit_node(self,X,top_n):
        N_visit = self.N
        childrens = [i for i in self.children]
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

    def single_rollout(self,X,rollout_round,board_uct,num_list=[5,1,1]):
        boards = []
        boards_rand = []
        for i in range(0, rollout_round): 
            self.do_rollout(board_uct)
            board_uct,board_rand = self.choose(board_uct)
            boards.append(list(board_uct.tup))
            boards_rand.append(list(board_rand))
        
        #visit nodes
        X_most_visit =  self.most_visit_node(X, num_list[1])
        
        #highest pred value nodes and random nodes
        new_x = self.data_process(X,boards)
        try:
            new_pred = self.model.predict(np.array(new_x).reshape(len(new_x),-1,1))
            new_pred = np.array(new_pred).reshape(len(new_x))
        except:
            pass
        boards_rand = np.vstack(boards_rand)
        new_rands = self.data_process(X,boards_rand)
        top_n = num_list[0]
        if len(new_x)>=top_n:
            ind = np.argsort(new_pred)[-top_n:]
            top_X =  new_x[ind]
            X_rand2 = [new_rands[random.randint(0, len(new_rands)-1)] for i in range(num_list[2])]
        elif len(new_x)==0:
            new_pred = self.model.predict(np.array(new_rands).reshape(len(new_rands),-1,1)).reshape(-1)
            ind = np.argsort(new_pred)[-top_n:]
            top_X =  new_rands[ind]
            X_rand2 = [new_rands[random.randint(0, len(new_rands)-1)] for i in range(num_list[2])]
        else:
            top_X = np.array(new_x)
            num_random = num_list[0] + num_list[2] - len(top_X)
            X_rand2 = [new_rands[random.randint(0, len(new_rands)-1)] for i in range(num_random)]
        try:
            top_X = np.concatenate([X_most_visit, top_X, X_rand2])
        except:
            top_X = np.concatenate([X_most_visit, top_X])
        
        return top_X
    
    def rollout(self, X,y,rollout_round,ratio,iteration):
        if self.name == 'rastrigin' or self.name == 'ackley' or self.name == 'levy':
            index_max = np.argmax(y)
            print(max(y))
            initial_X = X[index_max,:]
            values = self.model.predict(np.array(initial_X).reshape(1,-1,1))
            values = float(np.array(values).reshape(1))
            board_uct = opt_task(tup=tuple(initial_X), value=values, terminal=False)
            exp_weight = ratio * abs(max(y))
            self.exploration_weight = exp_weight
            ### starting rollout
            if self.name == 'rastrigin':
                num_list1=[18,2,0]
            else:
                num_list1=[15,3,2]
            top_X = self.single_rollout(X,rollout_round,board_uct,num_list=num_list1)
            
        
        else:
            if iteration % 100 < 80:
                UCT_low=False
            else:
                UCT_low=True

            #### make sure unique initial points
            ind = np.argsort(y)
            x_current_top = X[ind[-3:]]
            x_current_top = np.unique(x_current_top, axis = 0)
            i = 0
            while len(x_current_top) < 3:
                x_current_top=np.concatenate((x_current_top.reshape(-1,self.f.dims),X[ind[i-4]].reshape(-1,self.f.dims)), axis = 0)
                i-=1
                x_current_top = np.unique(x_current_top, axis = 0)
            
            ### starting rollout
            X_top=[]
            for i in range(3):
                initial_X = x_current_top[i]
                values = max(y)
                exp_weight = ratio * abs(values)
                if UCT_low ==True:
                    values = self.model.predict(np.array(initial_X).reshape(1,-1,1))
                    values = float(np.array(values).reshape(1))
                    exp_weight = ratio*0.5*values
                self.exploration_weight = exp_weight
                board_uct = opt_task(tup=tuple(initial_X), value=values, terminal=False)
                top_X = self.single_rollout(X,rollout_round,board_uct)
                X_top.append(top_X)
            
            top_X = np.vstack(X_top)
            top_X = top_X[:20]
        return top_X

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        count = 0
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
              return path
            unexplored = self.children[node] - self.children.keys()
            def evaluate(n):
              return n.value
            if count == 50:
                return path
              # return max(path, key=evaluate)
          
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
        self.children[node] = node.find_children(action, self.f,self.model)

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        # reward = node.reward()
        reward = node.reward(self.model)
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
        # print(f'node with max uct is:{uct_node}')
        return uct_node

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
    def find_children(board,action,f,model):
        if board.terminal:
            return set()
        turn = f.turn
        aaa = np.arange(f.lb[0], f.ub[0] + f.turn, f.turn).round(5)
        all_tup=[]
        for index in action:
            tup = list(board.tup)
            flip = random.randint(0,5)
            if   flip ==0:
              tup[index] += turn
            elif flip ==1:
                tup[index] -= turn
            elif flip ==2:
              for i in range(int(f.dims/5)):
                index_2 = random.randint(0, len(tup)-1)
                tup[index_2] = np.random.choice(aaa)
            elif flip ==3:
              for i in range(int(f.dims/10)):
                index_2 = random.randint(0, len(tup)-1)
                tup[index_2] = np.random.choice(aaa)
            elif flip ==4:
                tup[index] = np.random.choice(aaa)
            elif flip ==5:
                tup[index] = np.random.choice(aaa)
            tup[index] = round(tup[index],5)
            
            tup = np.array(tup)
            ind1 = np.where(tup>f.ub[0])[0]
            if len(ind1) > 0:
                tup[ind1] = f.ub[0]
            ind1 = np.where(tup<f.lb[0])[0]
            if len(ind1) > 0:
                tup[ind1] = f.lb[0]
            # print(tup)
            all_tup.append(tup)

        all_value = model.predict(np.array(all_tup).reshape(len(all_tup),f.dims,1))
        is_terminal=False
        return  {opt_task(tuple(t), v[0], is_terminal) for t, v in  zip(all_tup,all_value)}

    def reward(board,model):
        values = model.predict(np.array(board.tup).reshape(1,-1,1))
        values = float(np.array(values).reshape(1))
        return values
    def is_terminal(board):
        return board.terminal

#######################################################
'DOTS-Greedy'
#######################################################
class MCTS_Greedy:
    def __init__(self, f = None, dims = 20, model = None, name = None):
        self.f = f
        self.turn = f.turn
        self.dims = dims
        self.model = model
        self.name = name
        
    def choose(self, board):
        "Choose the best successor of node. (Choose a move in the game)"
        turn = self.turn
        aaa = np.arange(self.f.lb[0], self.f.ub[0] + self.f.turn, self.f.turn).round(5)
        nodes = []
        for index in range(self.dims):
            tup = np.array(board)
            flip = random.randint(0,5)
            if   flip ==0:
              tup[index] += turn
            elif flip ==1:
                tup[index] -= turn
            elif flip ==2:
              for i in range(int(self.dims/5)):
                index_2 = random.randint(0, len(tup)-1)
                tup[index_2] = np.random.choice(aaa)
            elif flip ==3:
              for i in range(int(self.dims/10)):
                index_2 = random.randint(0, len(tup)-1)
                tup[index_2] = np.random.choice(aaa)
            elif flip ==4:
                tup[index] = np.random.choice(aaa)
            elif flip ==5:
                tup[index] = np.random.choice(aaa)
            tup[index] = round(tup[index],5)
            ind1 = np.where(tup>self.f.ub[0])[0]
            if len(ind1) > 0:
                tup[ind1] = self.f.ub[0]
            ind1 = np.where(tup<self.f.lb[0])[0]
            if len(ind1) > 0:
                tup[ind1] = self.f.lb[0]
            nodes.append(tup)
        values = self.model.predict(np.array(nodes).reshape(len(nodes),-1,1))
        values = np.array(values).reshape(len(nodes))
        ind = np.argmax(values)
        node = nodes[ind]
        value = values[ind]
        return node, value, nodes
    
    def data_process(self,X,boards):
        new_x = []
        boards = np.array(boards)
        boards = np.unique(boards, axis=0)
        for i in boards:
          temp_x = np.array(i)
          same = np.all(temp_x==X, axis=1)
          has_true = any(same)
          if has_true == False:
            new_x.append(temp_x)
        new_x= np.array(new_x)
        print(f'unique number of boards: {len(new_x)}')
        return new_x
    
    def single_rollout(self,X, initial_X, rollout_round, top_n = 20):
        values = self.model.predict(np.array(initial_X).reshape(1,-1,1))
        x_current=np.array(initial_X)
        cu_Y=np.array(values).reshape(-1)

        boards=[]
        boards_rand=[]
        for i in range(rollout_round): 
            board,temp_Y,board_rand=self.choose(x_current)
            boards.append(board)
            boards_rand.append(board_rand)
            if temp_Y>cu_Y*1:
                  x_current = np.array(board)
                  cu_Y=np.array(temp_Y)

        new_x = self.data_process(X,boards)
        new_pred = self.model.predict(np.array(new_x).reshape(len(new_x),-1,1))
        new_pred = np.array(new_pred).reshape(len(new_x))
        
        if len(new_x)>=top_n:
            ind = np.argsort(new_pred)[-top_n:]
            top_X =  new_x[ind]
        else:
            boards_rand = np.vstack(boards_rand)
            new_rands = self.data_process(X,boards_rand)
            random_X=new_rands[np.random.choice(len(new_rands),size=top_n-len(new_x),replace=False)]
            top_X=np.concatenate((new_x, random_X),axis=0)
        return top_X 
        
    
    def rollout(self, X, y, rollout_round):
        if self.name == 'rastrigin' or self.name == 'ackley':
            index_max = np.argmax(y)
            initial_X = X[index_max,:]
            top_X = self.single_rollout(X, initial_X, rollout_round, top_n = 20)
            
        else:
            #### unique initial points
            ind = np.argsort(y)
            x_current_top = X[ind[-3:]]
            x_current_top = np.unique(x_current_top, axis = 0)
            i = 0
            while len(x_current_top) < 3:
                x_current_top=np.concatenate((x_current_top.reshape(-1,self.f.dims),X[ind[i-4]].reshape(-1,self.f.dims)), axis = 0)
                i-=1
                x_current_top = np.unique(x_current_top, axis = 0)
            
            X_top=[]
            for i in range (3):
              initial_X = x_current_top[i]
              top_X = self.single_rollout(X, initial_X, rollout_round, top_n = 7)
              X_top.append(top_X)
            
            top_X = np.vstack(X_top)
            top_X = top_X[-20:]
        return top_X
    


    

#######################################################
'DOTS-Episilon-Greedy'
#######################################################
class MCTS_eGreedy:
    def __init__(self, f = None, dims = 20, model = None, name = None):
        self.f = f
        self.turn = f.turn
        self.dims = dims
        self.model = model
        self.name = name
        
    def choose(self, board):
        "Choose the best successor of node. (Choose a move in the game)"
        turn = self.turn
        aaa = np.arange(self.f.lb[0], self.f.ub[0] + self.f.turn, self.f.turn).round(5)
        nodes = []
        for index in range(self.dims):
            tup = np.array(board)
            flip = random.randint(0,5)
            if   flip ==0:
              tup[index] += turn
            elif flip ==1:
                tup[index] -= turn
            elif flip ==2:
              for i in range(int(self.dims/5)):
                index_2 = random.randint(0, len(tup)-1)
                tup[index_2] = np.random.choice(aaa)
            elif flip ==3:
              for i in range(int(self.dims/10)):
                index_2 = random.randint(0, len(tup)-1)
                tup[index_2] = np.random.choice(aaa)
            elif flip ==4:
                tup[index] = np.random.choice(aaa)
            elif flip ==5:
                tup[index] = np.random.choice(aaa)
            tup[index] = round(tup[index],5)
            ind1 = np.where(tup>self.f.ub[0])[0]
            if len(ind1) > 0:
                tup[ind1] = self.f.ub[0]
            ind1 = np.where(tup<self.f.lb[0])[0]
            if len(ind1) > 0:
                tup[ind1] = self.f.lb[0]
            nodes.append(tup)
        values = self.model.predict(np.array(nodes).reshape(len(nodes),-1,1))
        values = np.array(values).reshape(len(nodes))
        #######epsilon greedy: 80% best and 20% random
        random_num=np.random.randint(0,10)
        if random_num<8:
            ind = np.argmax(values)
        else:
            ind=np.random.randint(0,len(values))
        node = nodes[ind]
        value = values[ind]
        return node, value, nodes
    
    def data_process(self,X,boards):
        new_x = []
        boards = np.array(boards)
        boards = np.unique(boards, axis=0)
        for i in boards:
          temp_x = np.array(i)
          same = np.all(temp_x==X, axis=1)
          has_true = any(same)
          if has_true == False:
            new_x.append(temp_x)
        new_x= np.array(new_x)
        print(f'unique number of boards: {len(new_x)}')
        return new_x
    
    def single_rollout(self,X, initial_X, rollout_round, top_n = 16, top_n2 = 4):
        values = self.model.predict(np.array(initial_X).reshape(1,-1,1))
        x_current=np.array(initial_X)
        cu_Y=np.array(values).reshape(-1)

        boards=[]
        boards_rand=[]
        for i in range(rollout_round): 
            board,temp_Y,board_rand=self.choose(x_current)
            boards.append(board)
            boards_rand.append(board_rand)
            if temp_Y>cu_Y*1:
                  x_current = np.array(board)
                  cu_Y=np.array(temp_Y)

        new_x = self.data_process(X,boards)
        new_pred = self.model.predict(np.array(new_x).reshape(len(new_x),-1,1))
        new_pred = np.array(new_pred).reshape(len(new_x))
        
        boards_rand = np.vstack(boards_rand)
        new_rands = self.data_process(X,boards_rand)
        
        if len(new_x)>=top_n:
            ind = np.argsort(new_pred)
            ind2=np.random.choice(len(new_rands),size=top_n2,replace=False)
            top_X=np.concatenate((new_x[ind[-top_n:]], new_rands[ind2]),axis=0)
        else:
            random_X=new_rands[np.random.choice(len(new_rands),size=top_n+top_n2-len(new_x),replace=False)]
            top_X=np.concatenate((new_x, random_X),axis=0)
        return top_X 
        
    
    def rollout(self, X, y, rollout_round):
        if self.name == 'rastrigin' or self.name == 'ackley':
            index_max = np.argmax(y)
            initial_X = X[index_max,:]
            top_X = self.single_rollout(X, initial_X, rollout_round)
            
        else:
            #### unique initial points
            ind = np.argsort(y)
            x_current_top = X[ind[-3:]]
            x_current_top = np.unique(x_current_top, axis = 0)
            i = 0
            while len(x_current_top) < 3:
                x_current_top=np.concatenate((x_current_top.reshape(-1,self.f.dims),X[ind[i-4]].reshape(-1,self.f.dims)), axis = 0)
                i-=1
                x_current_top = np.unique(x_current_top, axis = 0)
            
            X_top=[]
            for i in range (3):
              initial_X = x_current_top[i]
              top_X = self.single_rollout(X, initial_X, rollout_round, top_n = 6, top_n2 = 1)
              X_top.append(top_X)
            
            top_X = np.vstack(X_top)
            top_X = top_X[-20:]
        return top_X
        

#######################################################
'Dual Annealing'
#######################################################
class DualAnnealing:
    def __init__(self, f = None, dims = 20, model = None, name = None):
        self.f = f
        self.turn = f.turn
        self.dims = dims
        self.model = model
        self.name = name
        self.mode = None
        self.all_proposed=[]
        
    def data_process(self,X,boards):
        new_x = []
        boards = np.array(boards)
        boards = np.unique(boards, axis=0)
        for i in boards:
          temp_x = np.array(i)
          same = np.all(temp_x==X, axis=1)
          has_true = any(same)
          if has_true == False:
            new_x.append(temp_x)
        new_x= np.array(new_x)
        print(f'unique number of boards: {len(new_x)}')
        return new_x
    
    def predict(self,x):
        x=np.round(x,int(-np.log10(self.f.turn)))
        self.all_proposed.append(x)
        try:
           pred = self.model.predict(np.array(x).reshape(len(x),self.f.dims,1))
           pred = np.array(pred).reshape(len(x))
        except:
           pred = self.model.predict(np.array(x).reshape(1,self.f.dims,1))
           pred = np.array(pred).reshape(1)
        if self.name == 'ackley':
            pred_fun=100/pred-0.01
        elif self.name == 'rastrigin':
            pred_fun = -1 * pred
        elif self.name == 'rosenbrock':
            pred_fun=(100/pred-0.01)*self.f.dims*100
        elif self.name == 'levy':
            pred_fun=(100/pred-0.01)*self.f.dims
        elif self.name == 'schwefel':
            pred_fun=(100/pred-0.01)*self.f.dims
        elif self.name == 'michalewicz':
            pred_fun = 100/pred
        elif self.name == 'griewank':
            pred_fun=(100/pred-0.01)*self.f.dims*0.1
        else:
            pred_fun=100/pred
        return pred_fun
    
    def single_rollout(self,X, x_current, rollout_round, top_n = 16, top_n2 = 4):
        from scipy.optimize import dual_annealing
        bounds = []
        for idx in range(0, len(self.f.lb) ):
            bounds.append( ( float(self.f.lb[idx]), float(self.f.ub[idx])) )

        if self.mode == 'fast':
            ret = dual_annealing(self.predict, bounds = bounds,maxfun = rollout_round,initial_temp=0.05,x0=x_current)
        elif self.mode == 'origin':
            ret = dual_annealing(self.predict, bounds = bounds,x0=x_current)
        self.all_proposed.append(np.round(ret.x,int(-np.log10(self.f.turn))))

        new_x = self.data_process(X,self.all_proposed)
        new_pred = self.model.predict(np.array(new_x).reshape(len(new_x),-1,1))
        new_pred = np.array(new_pred).reshape(len(new_x))
        
        try:
            ind = np.argsort(new_pred)
            new_x2 = new_x[ind[:-top_n]]
            ind2=np.random.choice(len(new_x2),size=top_n2,replace=False)
            top_X=np.concatenate((new_x[ind[-top_n:]], new_x2[ind2]),axis=0)
        except:
            aaa = np.arange(self.f.lb[0], self.f.ub[0] + self.f.turn, self.f.turn).round(5)
            random_X = np.random.choice(aaa,size=(top_n - len(new_x), self.f.dims))
            top_X=np.concatenate((new_x, random_X),axis=0)
        return top_X 
        
    
    def rollout(self, X, y, rollout_round):
        if self.name == 'ackley':
            rollout_round = 1000
        
        if self.name == 'rastrigin' or self.name == 'ackley':
            index_max = np.argmax(y)
            initial_X = X[index_max,:]
            top_X = self.single_rollout(X, initial_X, rollout_round)
            
        else:
            #### unique initial points
            ind = np.argsort(y)
            x_current_top = X[ind[-3:]]
            x_current_top = np.unique(x_current_top, axis = 0)
            i = 0
            while len(x_current_top) < 3:
                x_current_top=np.concatenate((x_current_top.reshape(-1,self.f.dims),X[ind[i-4]].reshape(-1,self.f.dims)), axis = 0)
                i-=1
                x_current_top = np.unique(x_current_top, axis = 0)
            
            X_top=[]
            for i in range(3):
              initial_X = x_current_top[i]
              top_X = self.single_rollout(X, initial_X, rollout_round, top_n = 6, top_n2 = 1)
              X_top.append(top_X)
            
            top_X = np.vstack(X_top)
            top_X = top_X[-20:]
        return top_X


#######################################################
'Differential Evolution'
#######################################################
class DifferentialEvolution:
    def __init__(self, f = None, dims = 20, model = None, name = None):
        self.f = f
        self.turn = f.turn
        self.dims = dims
        self.model = model
        self.name = name
        self.mode = None
        self.all_proposed=[]
        
    def data_process(self,X,boards):
        new_x = []
        boards = np.array(boards)
        boards = np.unique(boards, axis=0)
        for i in boards:
          temp_x = np.array(i)
          same = np.all(temp_x==X, axis=1)
          has_true = any(same)
          if has_true == False:
            new_x.append(temp_x)
        new_x= np.array(new_x)
        print(f'unique number of boards: {len(new_x)}')
        return new_x
    
    def predict(self,x):
        x=np.round(x,int(-np.log10(self.f.turn)))
        self.all_proposed.append(x)
        try:
           pred = self.model.predict(np.array(x).reshape(len(x),self.f.dims,1))
           pred = np.array(pred).reshape(len(x))
        except:
           pred = self.model.predict(np.array(x).reshape(1,self.f.dims,1))
           pred = np.array(pred).reshape(1)
        if self.name == 'ackley':
            pred_fun=100/pred-0.01
        elif self.name == 'rastrigin':
            pred_fun = -1 * pred
        elif self.name == 'rosenbrock':
            pred_fun=(100/pred-0.01)*self.f.dims*100
        elif self.name == 'levy':
            pred_fun=(100/pred-0.01)*self.f.dims
        elif self.name == 'schwefel':
            pred_fun=(100/pred-0.01)*self.f.dims
        elif self.name == 'michalewicz':
            pred_fun = 100/pred
        elif self.name == 'griewank':
            pred_fun=(100/pred-0.01)*self.f.dims*0.1
        else:
            pred_fun=100/pred
        return pred_fun
    
    def single_rollout(self,X, x_current, rollout_round, top_n = 16, top_n2 = 4):
        from scipy.optimize import differential_evolution
        bounds = []
        for idx in range(0, len(self.f.lb) ):
            bounds.append( ( float(self.f.lb[idx]), float(self.f.ub[idx])) )
        
        if self.mode == 'fast':
            popsize = int(max(100 / self.f.dims, 1))
            ret = differential_evolution(self.predict, bounds = bounds,x0=x_current,maxiter=1,popsize=popsize)
        elif self.mode == 'origin':
            ret = differential_evolution(self.predict, bounds = bounds,x0=x_current)
        self.all_proposed.append(np.round(ret.x,int(-np.log10(self.f.turn))))

        new_x = self.data_process(X,self.all_proposed)
        new_pred = self.model.predict(np.array(new_x).reshape(len(new_x),-1,1))
        new_pred = np.array(new_pred).reshape(len(new_x))
        
        try:
            ind = np.argsort(new_pred)
            new_x2 = new_x[ind[:-top_n]]
            ind2=np.random.choice(len(new_x2),size=top_n2,replace=False)
            top_X=np.concatenate((new_x[ind[-top_n:]], new_x2[ind2]),axis=0)
        except:
            aaa = np.arange(self.f.lb[0], self.f.ub[0] + self.f.turn, self.f.turn).round(5)
            random_X = np.random.choice(aaa,size=(top_n - len(new_x), self.f.dims))
            top_X=np.concatenate((new_x, random_X),axis=0)
        return top_X 
        
    
    def rollout(self, X, y, rollout_round):
        if self.name == 'rastrigin' or self.name == 'ackley':
            index_max = np.argmax(y)
            initial_X = X[index_max,:]
            top_X = self.single_rollout(X, initial_X, rollout_round)
            
        else:
            #### unique initial points
            ind = np.argsort(y)
            x_current_top = X[ind[-3:]]
            x_current_top = np.unique(x_current_top, axis = 0)
            i = 0
            while len(x_current_top) < 3:
                x_current_top=np.concatenate((x_current_top.reshape(-1,self.f.dims),X[ind[i-4]].reshape(-1,self.f.dims)), axis = 0)
                i-=1
                x_current_top = np.unique(x_current_top, axis = 0)
            
            X_top=[]
            for i in range(3):
              initial_X = x_current_top[i]
              top_X = self.single_rollout(X, initial_X, rollout_round, top_n = 6, top_n2 = 1)
              X_top.append(top_X)
            
            top_X = np.vstack(X_top)
            top_X = top_X[-20:]
        return top_X


#######################################################
'CMA-ES'
#######################################################
class CMAES:
    def __init__(self, f = None, dims = 20, model = None, name = None):
        self.f = f
        self.turn = f.turn
        self.dims = dims
        self.model = model
        self.name = name
        self.mode = None
        self.all_proposed=[]
        
    def data_process(self,X,boards):
        new_x = []
        boards = np.array(boards)
        boards = np.unique(boards, axis=0)
        for i in boards:
          temp_x = np.array(i)
          same = np.all(temp_x==X, axis=1)
          has_true = any(same)
          if has_true == False:
            new_x.append(temp_x)
        new_x= np.array(new_x)
        print(f'unique number of boards: {len(new_x)}')
        return new_x
    
    def predict(self,x):
        x=np.round(x,int(-np.log10(self.f.turn)))
        self.all_proposed.append(x)
        try:
           pred = self.model.predict(np.array(x).reshape(len(x),self.f.dims,1))
           pred = np.array(pred).reshape(len(x))
        except:
           pred = self.model.predict(np.array(x).reshape(1,self.f.dims,1))
           pred = np.array(pred).reshape(1)
        if self.name == 'ackley':
            pred_fun=100/pred-0.01
        elif self.name == 'rastrigin':
            pred_fun = -1 * pred
        elif self.name == 'rosenbrock':
            pred_fun=(100/pred-0.01)*self.f.dims*100
        elif self.name == 'levy':
            pred_fun=(100/pred-0.01)*self.f.dims
        elif self.name == 'schwefel':
            pred_fun=(100/pred-0.01)*self.f.dims
        elif self.name == 'michalewicz':
            pred_fun = 100/pred
        elif self.name == 'griewank':
            pred_fun=(100/pred-0.01)*self.f.dims*0.1
        else:
            pred_fun=100/pred
        return pred_fun
    
    def single_rollout(self,X, x_current, rollout_round, top_n = 16, top_n2 = 4):
        import cma
        bounds = []
        for idx in range(0, len(self.f.lb) ):
            bounds.append( ( float(self.f.lb[idx]), float(self.f.ub[idx])) )
        
        if self.mode == 'fast':
            options = {'maxiter':int(rollout_round/10),'bounds':[self.f.lb[0], self.f.ub[0]]}
            es =cma.fmin(self.predict, x_current, 0.5, options)
        elif self.mode == 'origin':
            options = {'bounds':[self.f.lb[0], self.f.ub[0]]}
            es =cma.fmin(self.predict, x_current, (self.f.ub[0]-self.f.lb[0])/4, options)

        new_x = self.data_process(X,self.all_proposed)
        new_pred = self.model.predict(np.array(new_x).reshape(len(new_x),-1,1))
        new_pred = np.array(new_pred).reshape(len(new_x))
        
        try:
            ind = np.argsort(new_pred)
            new_x2 = new_x[ind[:-top_n]]
            ind2=np.random.choice(len(new_x2),size=top_n2,replace=False)
            top_X=np.concatenate((new_x[ind[-top_n:]], new_x2[ind2]),axis=0)
        except:
            aaa = np.arange(self.f.lb[0], self.f.ub[0] + self.f.turn, self.f.turn).round(5)
            random_X = np.random.choice(aaa,size=(top_n - len(new_x), self.f.dims))
            top_X=np.concatenate((new_x, random_X),axis=0)
        return top_X 
        
    
    def rollout(self, X, y, rollout_round):
        if self.name == 'rastrigin' or self.name == 'ackley':
            index_max = np.argmax(y)
            initial_X = X[index_max,:]
            top_X = self.single_rollout(X, initial_X, rollout_round)
            
        else:
            #### unique initial points
            ind = np.argsort(y)
            x_current_top = X[ind[-3:]]
            x_current_top = np.unique(x_current_top, axis = 0)
            i = 0
            while len(x_current_top) < 3:
                x_current_top=np.concatenate((x_current_top.reshape(-1,self.f.dims),X[ind[i-4]].reshape(-1,self.f.dims)), axis = 0)
                i-=1
                x_current_top = np.unique(x_current_top, axis = 0)
            
            X_top=[]
            for i in range(3):
              initial_X = x_current_top[i]
              top_X = self.single_rollout(X, initial_X, rollout_round, top_n = 6, top_n2 = 1)
              X_top.append(top_X)
            
            top_X = np.vstack(X_top)
            top_X = top_X[-20:]
        return top_X



