import numpy as np
from perceptron import Perceptron

training_inputs = []
training_inputs.append(np.array([1,1]))
training_inputs.append(np.array([0,1]))
training_inputs.append(np.array([1,0]))
training_inputs.append(np.array([0,0]))

labels = np.array([1,0,0,0])

perceptron = Perceptron(2)
perceptron.train(training_inputs,labels)



inputs = np.array([1,1])
print(perceptron.predict(inputs))

from time import process_time
# assigning n = 50  
n = 50 
  
# Start the stopwatch / counter  
t1_start = process_time()  
   
for i in range(n): 
    print(i, end =' ') 
  
print()  
  
# Stop the stopwatch / counter 
t1_stop = process_time() 
   
print("Elapsed time:", t1_stop, t1_start)  
   
print("Elapsed time during the whole program in seconds:", 
                                         t1_stop-t1_start)