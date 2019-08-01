"""
basic.py
~~~

Basic implementation of feedforward neural network with stochastic gradient 
descent. Refers to Nielson's neural networks and deep learning book
"""

import random
import numpy as np 

# helper functions
def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

class Network(object):

  def __init__(self, layer_sizes):
    '''layer_sizes is a list such as [2, 3, 1] that gives the number of 
    neurons in each layer. In this case, there would be 2 inputs, 3 hidden 
    neurons, and 1 output neuron. Weights and biases are initialized randomly 
    here using a Normal distribution

    The biases would look like: [[[randnum], 
                                  [randnum], 
                                  [randnum]], [[randnum]]]

    And the weights: [[[randnum, randnum], 
                        [randnum, randnum], 
                        [randnum, randnum]], [[randnum, randum, randum]]]

    where each element in the outer array [] represents a layer, each row 
    inside each element represents a neuron, and each element inside each row 
    represents a weight per input (biases have one)
    '''

    self.num_layers = len(layer_sizes)
    self.sizes = layer_sizes
    self.biases = [np.random.randn(neurons, 1) for neurons in layer_sizes[1:]]
    self.weights = [np.random.randn(neurons, inputs) for neurons, inputs
                     in zip(layer_sizes[1:], layer_sizes[:-1])]

  def feedforward(self, a):
    '''Returns output of a network: a = sigmoid(wa + b)'''
    for weights, biases in zip(self.weights, self.biases):
      # we are now in a single layer
      a = sigmoid(np.dot(weights, a) + biases)
    return a

  def stochastic_gradient_descent(self, training_data, learning_rate, epochs,
                                   mini_batch_size, test_data=None):
    '''Train the network using training_data (list of (x,y)) divided into 
    ``epochs`` number of epochs of size ``mini_batch_size``. If test_data is
    provided then the netework will be partially evaluated at each epoch. 
    The gradient descent is done at ```learning_rate```.'''
    
    training_data = list(training_data)
    n = len(training_data)

    for i in range(epochs):
      # within one epoch, or stage of training, divide the batches
      random.shuffle(training_data)
      # segment training_data into batches for SGD
      mini_batches = [training_data[k:k + mini_batch_size]
                     for k in range(0, n, mini_batch_size)]
      for mini_batch in mini_batches:
        # the actual gradient descent / updating part
        self.update_network(mini_batch, learning_rate)

      print('Epoch {} out of {} complete'.format(i, epochs))
      if test_data:
        test_data = list(test_data)
        n_test = len(test_data)
        print('Epoch {} Evaluation: {} / {} Success'.format(i, 
              self.evaluate(test_data), n_test))
    return

  def update_network(self, mini_batch, learning_rate):
    return

  def evaluate(self, test_data):
    return