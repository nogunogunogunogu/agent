from collections import OrderedDict
import random
from tkinter import *

from sklearn.metrics import f1_score
import theano

import numpy as np
import theano.tensor as T


SIMULATION_LENGTH = 1000

WORLD_CELL_NUM = 10
WORLD_CELL = 80
WORLD_SCALE = WORLD_CELL * WORLD_CELL_NUM
MOVE_SCALE = 2

def get_hurm_score(num_of_neighbor):
    if num_of_neighbor <= 2:
        return 100
    if num_of_neighbor <= 4:    
        return 40
    if num_of_neighbor <= 7:    
        return -100
    else:    
        return -200

# Stochastic Gradient Descent
def sgd(params, g_params, eps=np.float32(0.1)):
    updates = OrderedDict()
    for param, g_param in zip(params, g_params):
        updates[param] = param - eps*g_param
    return updates   

def get_near_adresses(address):
    nears = [[i, j] for i in range(address[0] - 1, address[0] + 2) 
             for j in range(address[1] - 1, address[1] + 2)]
    return nears
    
def move_limitation(coo):
    """
    if coo[0] < 0:
        coo[0] = 0
    if coo[0] > WORLD_SCALE:
        coo[0] = WORLD_SCALE
    if coo[1] < 0:
        coo[1] = 0
    if coo[1] > WORLD_SCALE:
        coo[1] = WORLD_SCALE   
    return coo
    """
    return coo%WORLD_SCALE

class voice():
    def __init__(self, words):
        self.information = [] * 2

rng = np.random.RandomState(1234)
class Agent():
    def __init__(self):
        self.coo = np.random.uniform(low=0, high=WORLD_SCALE, size=2)
        self.address = (self.coo/WORLD_CELL).astype('int32')
        self.degree = 0
        self.score = 0
    def get_coo(self):
        return self.coo
    def get_address(self):
        return self.address
    def update_address(self):
        self.address = (self.coo/WORLD_CELL).astype('int32')
    def get_point(self):
        return self.score
    def hurmed(self, agents):
        self.score -= get_hurm_score(self.count_friends(agents))
    def count_friends(self,agents):
        count = -1
        for agent in agents:
            for address in get_near_adresses(agent.address):
                if all(address == self.address):
                    count += 1
        return count    
          
    def invoice(self, invoice):
        return
        
    def outvoice(self):
        return
    
    def move(self, degree, scale): # scale must be [0, 1]
        self.degree += degree
        if scale < 0:
            scale = 0
        elif scale > 1:
            scale = 1
        self.coo = self.coo + np.array([MOVE_SCALE * scale * np.cos(self.degree), 
                                        MOVE_SCALE * scale * np.sin(self.degree)])
        self.update_address()
        self.coo = move_limitation(self.coo)
        
    def see(self, agents, enemies):
        sight = [0 for i in range(9)]
        for enemy in enemies:
            for i, address in enumerate(get_near_adresses(enemy.address)):
                if all(address == self.address):
                    sight[i] += 1
        return sight


class Enemy():
    def __init__(self):
        self.coo = np.random.uniform(low=0, high=WORLD_SCALE, size=2)
        self.address = (self.coo/WORLD_CELL).astype('int32')
        self.degree = 0
        self.score = 0
    def get_coo(self):
        return self.coo
    def get_point(self):
        return self.score
    def update_address(self):
        self.address = (self.coo/WORLD_CELL).astype('int32')
    def get_address(self):
        return self.address
    
    class Layer:
        # Constructor
        def __init__(self, in_dim, out_dim, function):
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.function = function
            self.W = theano.shared(rng.uniform(low=-0.08, high=0.08,size=(in_dim, out_dim)).astype('float32'), 
                                   name='W')
            self.b = theano.shared(np.zeros(out_dim).astype('float32'), name='b')
            self.params = [self.W, self.b]
    
        # Forward Propagation
        def f_prop(self, x):
            self.z = self.function(T.dot(x, self.W) + self.b)
            return self.z 
    x = T.fvector('x')
    t = T.ivector('t')
    
    layers = [
#         Layer(9, 6, T.nnet.sigmoid),
        Layer(9, 8, T.nnet.softmax)
    ]
    
    params = []
    for i, layer in enumerate(layers):
        params += layer.params
        if i == 0:
            layer_out = layer.f_prop(x)
        else:
            layer_out = layer.f_prop(layer_out)
    
    y = layers[-1].z
    cost = T.mean(T.nnet.categorical_crossentropy(y, t))
    
    g_params = T.grad(cost=cost, wrt=params)
    updates = sgd(params, g_params)
    
    train = theano.function(inputs=[x, t], outputs=cost, updates=updates, allow_input_downcast=True, 
                            name='train')
    test = theano.function(inputs=[x], outputs=T.argmax(y, axis=1), name='test')
    
    
    def train_weight(self):
        return 0
        
    
    def move(self, agents):
        degree = np.pi / 4
        scale = 1
        self.degree = degree * self.test(self.see(agents))[0]
        self.coo = self.coo + np.array([MOVE_SCALE * scale * np.cos(self.degree), 
                                        MOVE_SCALE * scale * np.sin(self.degree)])
        self.update_address()
        self.coo = move_limitation(self.coo)
        
    def see(self, agents):
        sight = [0 for i in range(9)]
        for agent in agents:
            for i, address in enumerate(get_near_adresses(agent.address)):
                if all(address == self.address):
                    sight[i] += 1
        return sight
                
    def hurm(self,agents):
        for agent in agents:
            if all(agent.get_address() == self.address):
                agent.hurmed(agents)
                self.score += get_hurm_score(agent.count_friends(agents))
               
agents = [Agent() for i in range(50)]
enemies = [Enemy() for i in range(2)]

pos = []
for i in range(SIMULATION_LENGTH):
    pos_in=[]
    for agent in agents:
        agent.invoice(0)
        agent.outvoice()
        agent.move(0.1, 1)
        pos_in.append(agent.get_coo())
    for enemy in enemies:
        enemy.move(agents)    
        pos_in.append(enemy.get_coo())
        enemy.hurm(agents)
        if i % 100 == 0:
            enemy.train_weight()
    pos.append(pos_in)
    

cnt = 0 
root = Tk()
c0 = Canvas(root, width = WORLD_SCALE, height = WORLD_SCALE)
c0.pack(expand = True, fill = BOTH)
for i in range(-1, WORLD_CELL_NUM):
    c0.create_line(0,i * WORLD_CELL,800,i * WORLD_CELL)
    c0.create_line(i * WORLD_CELL,0,i * WORLD_CELL,800)

ovals = [c0.create_oval(0,0,0,0,fill = 'red') for i in range(len(agents))]
ovals2 = [c0.create_oval(0,0,0,0,outline = 'blue') for i in range(len(enemies))]

OUTPUT_OVAL = 2
def draw_oval(cnt):
    global pos
    [c0.coords(ovals[i],pos[cnt][i][0]-OUTPUT_OVAL,pos[cnt][i][1]-OUTPUT_OVAL,pos[cnt][i][0]+OUTPUT_OVAL,
               pos[cnt][i][1]+OUTPUT_OVAL) for i in range(len(agents))]
    [c0.coords(ovals2[i],pos[cnt][i+len(agents)][0]-OUTPUT_OVAL,pos[cnt][i+len(agents)][1]-OUTPUT_OVAL,
               pos[cnt][i+len(agents)][0]+OUTPUT_OVAL,pos[cnt][i+len(agents)][1]+OUTPUT_OVAL) 
     for i in range(len(enemies))]
        
def show():
    global cnt
    if cnt == SIMULATION_LENGTH:
        return
    draw_oval(cnt)
    root.after(30, show)
    cnt += 1



print("points of agents:")
for agent in agents:
    print("%5d,"  % agent.get_point(), end=" ")
print("")
    
print("points of enemies:")
for enemy in enemies:
    print("%5d,"  % enemy.get_point(), end=" ")

show()
root.mainloop()  
