"""
basic.py
~~~

Basic implementation of feedforward neural network with stochastic gradient descent. Refers to 
Nielson's neural networks and deep learning book
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

    where each element in the outer array [] represents a layer, each row inside each element
    represents a neuron, and each element inside each row represents a weight per input (biases have one)
    '''

    self.num_layers = len(layer_sizes)
    self.sizes = layer_sizes
    self.biases = [np.random.randn(neurons, 1) for neurons in layer_sizes[1:]]
    self.weights = [np.random.randn(neurons, inputs) for neurons, inputs in zip(layer_sizes[1:], layer_sizes[:-1])]

  def feedforward(self, a):
    '''Returns output of a network: a = sigmoid(wa + b)'''

    for weights, biases in zip(self.weights, self.biases):
      # we are now in a single layer
      a = sigmoid(np.dot(weights, a) + biases)
    return a
