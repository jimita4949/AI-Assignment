# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 22:28:12 2020

@author: JIMM
"""

import numpy as np;
X=[0.5,2.5,5,8,9]
Y=[0.2,0.9,1,3,4]

def f(w,b,x):
    return 1.0/(1.0+np.exp(-(w*x+b)))

def error(w,b):
    err=0.0
    for x,y in zip(X,Y):    
        fx=f(w,b,x)
        err+=0.5*(fx-y)**2
    return err
   

def grad_b(w,b,x,y):
    fx=f(w,b,x)
    return (fx-y)*fx*(1-fx)

def grad_w(w,b,x,y):
    fx=f(w,b,x)
    return (fx-y)*fx*(1-fx)*x


            
def do_momentum_gradient_descent():
    w, b, eta= 5,5,1.0
    init_w=-2
    init_b=-2
    max_epochs = 1000
    prev_v_w, prev_v_b, gamma = 0, 0, 0.9
    for i in range(max_epochs):
        dw, db = 0,0
        
        for x,y in zip(X,Y): 
              dw+=grad_w(w,b,x,y)
              db+=grad_b(w,b,x,y)    
        v_w = gamma * prev_v_w + eta * dw
        v_b = gamma * prev_v_b + eta * db
        w = w - v_w
        b = b - v_b
        prev_v_w = v_w
        prev_v_b = v_b
    print("momentum gradient descent")
    print("-------------------")
    print("2weight")
    print(w)
    print("2bias")
    print(b)
    print("error")
    print(error(w, b))
        
    

do_momentum_gradient_descent()

