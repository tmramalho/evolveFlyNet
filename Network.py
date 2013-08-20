'''
Created on Aug 16, 2013

@author: tiago
'''

import numpy as np
import theano
import theano.tensor as T

class Layer(object):
	'''Based on the feedforward net from Theano docs '''
	def __init__(self, rng, inputVec, n_in, n_out, W=None, b=None,
				 activation=T.tanh):
		self.input = input
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

		lin_output = T.dot(inputVec, self.W) + self.b
		self.output = (lin_output if activation is None
					   else activation(lin_output))
		# parameters of the model
		self.params = [self.W, self.b]
		
	def cost(self, target):
		return ((self.output - target)**2).sum()
		
class Network(object):
	def __init__(self, rng, structure, inputSize, inputVec):
		self.layers = []
		
		for i,_ in enumerate(structure):
			if i == 0:
				layer = Layer(rng, inputVec, inputSize, structure[i])
			else:
				layer = Layer(rng, self.layers[i-1].output, structure[i-1], structure[i])
			self.layers.append(layer)
		
		self.params = []
		self.L1 = 0
		self.L2_sqr = 0
		
		for l in self.layers:
			self.params += l.params
			self.L1 += abs(l.W).sum()
			self.L2_sqr += (l.W ** 2).sum()
		
		self.input = inputVec
		#self.output = self.layers[-1].output
		self.output = theano.printing.Print('no')(self.layers[-1].output)
	
	def outputCost(self, target):
		return self.layers[-1].cost(target)
	
	def setInput(self, inputVec):
		self.input.set_value(inputVec)
	
if __name__ == '__main__':
	rng = np.random.RandomState(1234)
	x = theano.shared(np.ones(5))
	n = Network(rng, [8,2], 5, x)
	f = theano.function([], n.output)
	print f()
	n.setInput(np.array([1,2,3,4,5]))
	print f()
	target = theano.shared(np.ones(2))
	oc = n.outputCost(target)
	h = theano.function([], oc)
	print h()