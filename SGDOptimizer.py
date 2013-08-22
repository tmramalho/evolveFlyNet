'''
Created on Aug 22, 2013

@author: tiago
'''

import numpy as np
import theano
import theano.tensor as T
import Network as net
import Integrate as ig

class SGDOptimizer(object):
	'''
	Fit a network with minibatch stochastic gradient descent
	(batch size must be set to one for now due to network)
	'''

	def __init__(self, inSamples, outSamples, c0, learning_rate = 0.1, seed = 1234, L1_reg = 0, L2_reg = 0):
		'''
		Constructor
		'''
		rng = np.random.RandomState(seed)
		n = net.Network(rng, [8,2], 5)
		o = ig.ODESolver(n)
		index = T.lscalar()
		inputSequence = T.fmatrix("is")
		outputSequence = T.fmatrix("os")
		initialState = T.fvector("c_i")
		integ = ig.Integrate(inputSequence, outputSequence, initialState, o.eulerStep)
		
		cost = integ.score + L1_reg * n.L1 + L2_reg * n.L2_sqr
				
		gparams = []
		for param in n.params:
			gparam = T.grad(cost, param)
			gparams.append(gparam)
	
		updates = []
		for param, gparam in zip(n.params, gparams):
			updates.append((param, param - learning_rate * gparam))
		
		print "Compiling model..."
		self.model = theano.function(inputs=[index],
				outputs=cost,
				updates=updates + integ.updates,
				givens={
					inputSequence: inSamples[index],
					outputSequence: outSamples[index],
					initialState: c0})
		
	def trainModel(self):
		for _ in range(0,50):
			for index in range(0,50):
				score = self.model(index)
				print score
		
if __name__ == '__main__':
	rng = np.random.RandomState(1234)
	iSeq = np.array(rng.rand(300,5), dtype='float32')
	oSeq = np.array([[1,1],[0.2,0.3],[0.4,0.4]], dtype='float32')
	c0 = theano.shared(np.array([0.1,0.1], dtype='float32'))
	iSamples = theano.shared(np.tile(iSeq, (50,1,1)))
	oSamples = theano.shared(np.tile(oSeq, (50,1,1)))
	sgd = SGDOptimizer(iSamples, oSamples, c0)
	print "Dry run!"
	sgd.trainModel()