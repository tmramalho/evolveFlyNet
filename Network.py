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
		self.rng = rng
		self.n_in = n_in
		self.n_out = n_out
		self.activation = activation
			
	def run(self, inputVec):
		"""
		Compute this layer's output
		
		Keyword arguments:
		inputVec -- vector with the input values
		"""
		lin_output = T.dot(self.inputFunction(inputVec), self.W) + self.b
		return (lin_output if self.activation is None
					   else self.activation(lin_output))
	
	def reset(self):
		W_values = np.asarray(self.rng.uniform(
				low=-np.sqrt(6. / (self.n_in + self.n_out)),
				high=np.sqrt(6. / (self.n_in + self.n_out)),
				size=(self.n_in, self.n_out)), dtype=theano.config.floatX)
		if self.activation == theano.tensor.nnet.sigmoid:
			W_values *= 4

		self.W.set_value(W_values)

		b_values = np.zeros((self.n_out,), dtype=theano.config.floatX)
		self.b.set_value(b_values)
		
class Network(object):
	def __init__(self, rng, structure, inputSize, activation = T.tanh):
		"""
		Initialize the network by constructing the needed layer objects
		and connecting their outputs and inputs.
		Construct the expressions representative of the parameters and
		regularization costs needed for the optimization
	
		Keyword arguments:
		rng -- the random number generator (for unspecified parameters)
		structure -- list with number of outputs for each layer
		inputSize -- dimension of input for first layer
		"""
		self.layers = []
		
		for i,_ in enumerate(structure):
			if i == 0:
				layer = Layer(rng, lambda x: x,          inputSize,      structure[i], activation = activation)
			else:
				layer = Layer(rng, self.layers[i-1].run, structure[i-1], structure[i], activation = activation)
			self.layers.append(layer)
		
		self.L1 = 0
		self.L2_sqr = 0
		self.structure = structure
		self.structure.insert(0, inputSize)
		self.createParameterList()
		
	def createParameterList(self):
		self.params = []
		for l in self.layers:
			self.params += [l.W, l.b]
			self.L1 += abs(l.W).sum()
			self.L2_sqr += (l.W ** 2).sum()
		
	def run(self, inputVec):
		"""
		Returns the output of the very last layer
		
		Keyword arguments:
		inputVec -- vector with the input values
		"""
		return self.layers[-1].run(inputVec)
	
	def reset(self):
		for l in self.layers:
			l.reset()
	
def testSingleSample():
	rng = np.random.RandomState(1234)
	x = T.fvector('x')
	x0 = theano.shared(np.ones(5, dtype='float32'))
	n = Network(rng, [8,2], 5)
	f = theano.function([], n.run(x), givens={x:x0})
	print f()
	x0.set_value(np.array([1,2,3,4,5], dtype='float32'))
	print f()
	
def testMultiSample():
	rng = np.random.RandomState(1234)
	x = T.fmatrix('x')
	x0 = theano.shared(np.tile(np.ones(5, dtype='float32'),(100,1)))
	n = Network(rng, [8,2], 5)
	f = theano.function([], n.run(x), givens={x:x0})
	print f()

if __name__ == '__main__':
	testMultiSample()
	