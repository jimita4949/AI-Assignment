# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 12:57:38 2020

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

def do_gradient_descent():
    w,b,eta,max_epochs=-2,-2,1.0,10
    for i in range(max_epochs):
        dw,db=0,0
        for x,y in zip(X,Y):
            dw,db=0,0
            for x,y in zip(X,Y):
                dw+=grad_w(w,b,x,y)
                db+=grad_b(w,b,x,y)
            w=w-eta*dw
            b=b-eta*db
            print("weight")
            print(w)
            print("bias")
            print(b)
    
do_gradient_descent()