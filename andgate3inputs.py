# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 22:07:53 2020

@author: JIMM
"""
import numpy as np
from perceptron import Perceptron

training_inputs = []

training_inputs.append(np.array([0,0,0]))
training_inputs.append(np.array([1,0,1]))
training_inputs.append(np.array([0,1,0]))
training_inputs.append(np.array([0,1,1]))
training_inputs.append(np.array([1,1,0]))
training_inputs.append(np.array([1,1,1]))

#predict for rest 2
#training_inputs.append(np.array([0,0,1)) 
#training_inputs.append(np.array([1,0,0]))

labels = np.array([0,0,0,0,0,1])

perceptron = Perceptron(3)
perceptron.train(training_inputs,labels)
print("prediction for 0,0,1")
inputs = np.array([0,0,1])
print(perceptron.predict(inputs))

print("prediction for 1,0,0")
inputs = np.array([1,0,0])
print(perceptron.predict(inputs))



