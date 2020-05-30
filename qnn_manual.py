# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 23:58:04 2020

@author: xialiqiao
"""
import numpy as np
import random
import math
from matplotlib import pyplot as plt

def Pauli_Y(i):
    return np.array([[0,-i],[i,0]])

def CNOT():
    return np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])

def RY(theta):
    return np.array([[math.cos(theta/2),-math.sin(theta/2)],[math.sin(theta/2),math.cos(theta/2)]])

def measure(vector):
    matrix0=np.array([[1,0],[0,0]])
    matrix1=np.array([[0,0],[0,1]])
    prob0=np.dot(np.dot(vector,matrix0),vector)
    prob1=np.dot(np.dot(vector,matrix1),vector)
    return prob0/(prob0+prob1),prob1/(prob0+prob1)

def forward(x,i):
    input_tmp=np.dot(x,RY(i))
    prob0,prob1=measure(input_tmp)
    if prob0>prob1:
        return 0,1-prob0
    else:
        return 1,prob1

def grad(data,theta,stride=0.1):
    up=forward(np.array(data),theta+stride)[1]
    down=forward(np.array(data),theta-stride)[1]
    grad=(up-down)/2
    return theta-grad

def cross_entropy(y_true,y_pred):
    return -y_true*math.log(y_pred)-(1-y_true)*math.log(1-y_pred)

####i=i+output*delta(y)

#####球坐标
n=100
X=np.array([(random.uniform(-1,1)*np.pi*2,random.uniform(-1,1)*np.pi*2) for t in range(n)])
Y=[random.randint(0,1) for t in range(n)]
epoch=10
init_theta=0.1
#y_pred=np.array([forward(x,0.2)[1] for x in X])
#y_true=np.array(Y)
#cost = (- 1 / n) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * (np.log(1 - y_pred)))
#d_theta=sum([grad(x,0.2,0.1) for x in X])/n
cost_collect=[]
for t in range(epoch):
    if t==0:
        theta_tmp=init_theta
    y_pred=np.array([forward(x,theta_tmp)[1] for x in X])
    y_true=np.array(Y)
    cost = (- 1 / n) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * (np.log(1 - y_pred)))
    d_theta=sum([grad(x,theta_tmp,0.05) for x in X])/n
    theta_tmp-=d_theta
    #theta_tmp=d_theta
    cost_collect.append(cost)

plt.plot(cost_collect)
plt.title("QNN cost")
plt.xlabel('Training Iterations')
plt.ylabel('cross_entropy')
plt.show()

plt.scatter(X,Y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()