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

def sigmoid_prime(x):
  return sigmoid(x) * (1 - sigmoid(x))

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
    '''Update the network's weights and biases using SGD and backpropogation
    mini batch is a list of training data in the form (x, y)'''
    # Cost function has the form 1/n sum(C_x) (average of all training example 
    # costs)
    del_b = [np.zeros(b.shape) for b in self.biases]
    del_w = [np.zeros(w.shape) for w in self.weights]
    for x, y in mini_batch:
      # the effect on the gradient from one particular training example
      marginal_del_b, marinal_del_w = self.backpropogation(x, y)
      # add this effect from each data point to each layer of the network
      del_b = [b + marginalb for b, marginalb in zip(del_b, marginal_del_b)]
      del_w = [w + marginalw for w, marginalw in zip(del_w, marginal_del_w)]
    # we have updated del_b and del_w for all the training examples in 
    # the batch. We can now use this to update the weights of the network
    # Remember, the learning rule is v = v - learning_rate*del_C
    # del_c is an avg over the batch, so we have v = v - eta/batch_size*del_C
    self.weights = [w - learning_rate/len(mini_batch)*dw 
                    for w, dw in zip(self.weights, del_w)]
    self.biases = [b - learning_rate/len(mini_batch)*db 
                    for b, db in zip(self.biases, del_b)]

    return

  def backpropagation(self, x, y):
    '''Returns a tuple (del_b, del_w) representing the gradient of the cost
    function. del_b and del_w are similar in structure to biases and weights, 
    each is a layer by layer numpy array'''
    del_b = [np.zeros(b.shape) for b in self.biases]
    del_w = [np.zeros(w.shape) for w in self.weights]

    # step 1: feedforward to compute activations a and z
    activation = x # x, the input, is the first layer of the network
    activations = [x] # each element is one layer in the network
    zs = [] # each element is one layer in the network
    for b, w in zip(self.biases, self.weights):
      z = np.dot(w, activation) + b
      activation = sigmoid(z)
      zs.append(z)
      activations.append(activation)

    # step 2: now that we have all the activations, we compute the output error
    # Recall the formula \delta_L = \del_A(C) \hadamard \sigmoid_prime(z_L)
    delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) 
    # Recall that \partialC / \partialb_j = \delta_j and that
    # \partialC / \partialw_j = a_{l-1} * \delta_l. We proved this through
    # chain rule
    del_b[-1] = delta
    del_w[-1] = np.dot(delta, activations[-2].transpose())

    # step 3: backpropogating the error starting from the output layer
    for l in range(2, self.num_layers):
      # l represents the layer # starting from the end
      # Recall \delta_l = w_{l+1}^T*\delta_{l+1} \hadamard \sigmoid_prime(z_L)
      z = zs[-l]
      delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(z)
      del_b[-l] = delta
      del_w[-l] = np.dot(delta, activations[-l].transpose())

    # we now have a set of gradients del_b and del_w to perform gradient descent
    return (del_b, del_w)

  def cost_derivative(self, output_activations, y):
    """Return the vector of partial derivatives (partial C_x) /
    (partial a) for each of the outputs in the output activation layer"""
    return (output_activations-y) # for a quadratic cost function


  def evaluate(self, test_data):
    '''Return the number of test inputs correctly guessed by model'''
    # the network's output is the index of whichever neuron in the final layer
    # has the highest activation. Hence, argmax()
    results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in results)