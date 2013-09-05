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

	def __init__(self, n, integ, inSamples, outSamples, c0, learning_rate = 0.1, L1_reg = 0, L2_reg = 0, batch_size = 20):
		'''
		Creates the cost function and gradient expressions, compiles the model
		
		Keyword arguments:
		inSamples -- the control variables to be fed in the integration loop (n_steps, n_samples, n_input_dims)
		outSamples -- the desired output (n_unit_steps, n_samples, n_output_dims)
		c0 -- initial state of the system (n_samples, n_output_dims)
		learning_rate -- multiplicative constant for the gradient
		l1_reg -- L1 regularization penalty
		l2_reg -- L2 regularization penalty
		batch_size -- number of samples to be evaluated simultaneously
		'''
		index = T.lscalar()
		inputSequence = T.ftensor3("is")
		outputSequence = T.ftensor3("os")
		initialState = T.fmatrix("c_i")
		integ.buildModel(inputSequence, outputSequence, initialState)
		
		cost = integ.mean + L1_reg * n.L1 + L2_reg * n.L2_sqr
				
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
					inputSequence: inSamples[:,index * batch_size:(index + 1) * batch_size,:],
					outputSequence: outSamples[:,index * batch_size:(index + 1) * batch_size,:],
					initialState: c0[index * batch_size:(index + 1) * batch_size]})
		
		self.eval = theano.function(inputs=[index],
				outputs=integ.cUnits,
				givens={
					inputSequence: inSamples[:,index * batch_size:(index + 1) * batch_size,:],
					outputSequence: outSamples[:,index * batch_size:(index + 1) * batch_size,:],
					initialState: c0[index * batch_size:(index + 1) * batch_size]})
		
def simpleTest():
	rng = np.random.RandomState(1234)
	iSeq = np.array(rng.rand(300,5), dtype='float32')
	oSeq = np.array([[1,1],[0.2,0.3],[0.4,0.4]], dtype='float32')
	c0 = np.array([0.1,0.1], dtype='float32')
	iSamples = theano.shared(np.rollaxis(np.tile(iSeq, (50,1,1)),0,2))
	oSamples = theano.shared(np.rollaxis(np.tile(oSeq, (50,1,1)),0,2))
	ci = theano.shared(np.tile(c0, (50,1)))
	print iSamples.shape.eval(), oSamples.shape.eval(), ci.shape.eval()
	n = net.Network(rng, [8,2], 5)
	o = ig.ODESolver(n)
	integ = ig.Integrate(o.eulerStep)
	bs = 1
	sgd = SGDOptimizer(n, integ, iSamples, oSamples, ci, batch_size=bs)
	print "Dry run!"
	for _ in range(0,50):
		for index in range(0,50/bs):
			score = sgd.model(index)
			print score

def combinedTest():
	rng = np.random.RandomState(1234)
	iSeq = np.array(rng.rand(300,3), dtype='float32')
	oSeq = np.array([[1,1],[0.2,0.3],[0.4,0.4]], dtype='float32')
	c0 = np.array([0.1,0.1], dtype='float32')
	iSamples = theano.shared(np.rollaxis(np.tile(iSeq, (50,1,1)),0,2))
	oSamples = theano.shared(np.rollaxis(np.tile(oSeq, (50,1,1)),0,2))
	ci = theano.shared(np.tile(c0, (50,1)))
	print iSamples.shape.eval(), oSamples.shape.eval(), ci.shape.eval()
	n = net.Network(rng, [8,2], 5)
	o = ig.ODESolver(n)
	integ = ig.Integrate(o.combinedEulerStep)
	bs = 1
	sgd = SGDOptimizer(n, integ, iSamples, oSamples, ci, batch_size=bs)
	print "Dry run!"
	for _ in range(0,50):
		for index in range(0,50/bs):
			score = sgd.model(index)
			print score
	
if __name__ == '__main__':
	combinedTest()