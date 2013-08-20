'''
Created on Aug 16, 2013

@author: tiago
'''

import numpy as np
import theano
import theano.tensor as T

class Layer(object):
	'''Based on the feedforward net from Theano docs '''
	def __init__(self, rng, inputFunction, n_in, n_out, W=None, b=None,
				 activation=T.tanh):
		"""
		Initialize the parameters for the layer
	
	    Keyword arguments:
	    rng -- the random number generator (for unspecified parameters)
	    inputFunction -- function applied to the input. If this is the 
	    first layer, it should be the identity. If this is layer i, should
	    be the output of layer i-1
	    n_in -- dim of input vector
	    n_out -- dim of output vector
	    W -- weight matrix
	    b -- bias matrix
	    activation -- activation function
	    """
		self.inputFunction = inputFunction
		if W is None:
			W_values = np.asarray(rng.uniform(
					low=-np.sqrt(6. / (n_in + n_out)),
					high=np.sqrt(6. / (n_in + n_out)),
					size=(n_in, n_out)), dtype=theano.config.floatX)
			if activation == theano.tensor.nnet.sigmoid:
				W_values *= 4

			W = theano.shared(value=W_values, name='W', borrow=True)

		if b is None:
			b_values = np.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)

		self.W = W
		self.b = b
		self.activation = activation

		# parameters of the model
		self.params = [self.W, self.b]
		
	def run(self, inputVec):
		lin_output = T.dot(self.inputFunction(inputVec), self.W) + self.b
		return (lin_output if self.activation is None
					   else self.activation(lin_output))
		
class Network(object):
	def __init__(self, rng, structure, inputSize):
		self.layers = []
		
		for i,_ in enumerate(structure):
			if i == 0:
				layer = Layer(rng, lambda x: x,          inputSize,      structure[i])
			else:
				layer = Layer(rng, self.layers[i-1].run, structure[i-1], structure[i])
			self.layers.append(layer)
		
		self.params = []
		self.L1 = 0
		self.L2_sqr = 0
		
		for l in self.layers:
			self.params += l.params
			self.L1 += abs(l.W).sum()
			self.L2_sqr += (l.W ** 2).sum()
		
	def run(self, inputVec):
		return self.layers[-1].run(inputVec)
	
if __name__ == '__main__':
	rng = np.random.RandomState(1234)
	x = T.fvector('x')
	x0 = theano.shared(np.ones(5, dtype='float32'))
	n = Network(rng, [8,2], 5)
	f = theano.function([], n.run(x), givens={x:x0})
	print f()
	x0.set_value(np.array([1,2,3,4,5], dtype='float32'))
	print f()