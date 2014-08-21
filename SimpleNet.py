'''
Created on Sep 7, 2013

@author: tiago
'''

import DataProcessor as dp
import SGDOptimizer as so
import Network as net
import Integrate as ig
import theano
import theano.tensor as T
import numpy as np
import scipy.interpolate as ip
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cPickle

class SimpleNet(object):
	'''
	classdocs
	'''
	
	def __init__(self, seed=1234):
		self.rng = np.random.RandomState(seed)
	
	def createNewNet(self, iDim, oDim, input_set, target_set, 
				activation = T.nnet.sigmoid, learning_rate = 0.1, batch_size = 20, L1_reg = 0, L2_reg = 0,
				nHidden = 10):
		'''
		Creates a new neural net object with the given structure and passes it to the compilation stage
		
		Keyword arguments:
		iDim -- number of input dimensions
		oDim -- number of output dimensions
		input_set -- theano shared variable containing input data
		target_set -- theano shared variable containing target data
		learning_rate -- multiplicative constant for the gradient
		l1_reg -- L1 regularization penalty
		l2_reg -- L2 regularization penalty
		batch_size -- number of samples to be evaluated simultaneously
		activation -- activation function
		nHidden -- number of hidden units
		'''
		if type(nHidden) == int:
			structure = [nHidden, oDim]
		elif type(nHidden) == list:
			structure = nHidden + [oDim]
		elif type(nHidden) == tuple:
			structure = list(nHidden) + [oDim]
		
		self.n = net.Network(self.rng, structure, iDim, activation = activation)
		self.buildNet(input_set, target_set, learning_rate, batch_size, L1_reg, L2_reg)
	
	def buildNet(self, input_set, target_set, learning_rate, batch_size, L1_reg, L2_reg):
		'''
		Creates the cost function and gradient expressions, compiles the model
		
		Keyword arguments:
		input_set -- theano shared variable containing input data
		target_set -- theano shared variable containing target data
		learning_rate -- multiplicative constant for the gradient
		l1_reg -- L1 regularization penalty
		l2_reg -- L2 regularization penalty
		batch_size -- number of samples to be evaluated simultaneously
		'''
		index = T.lscalar("ix")
		x = T.fmatrix('x')
		y = T.fmatrix('y')
		cost = T.mean((self.n.run(x)-y) ** 2) 
		regCost = cost + L1_reg * self.n.L1 + L2_reg * self.n.L2_sqr
		
		gparams = []
		for param in self.n.params:
			gparam = T.grad(regCost, param)
			gparams.append(gparam)
	
		updates = []
		for param, gparam in zip(self.n.params, gparams):
			updates.append((param, param - learning_rate * gparam))
			
		'''
		compute gradient w.r.t. inputs of the network
		supports batch evaluation (i.e. multiple samples simultaneously)
		'''
		xv = T.fvector('x')
		output = self.n.run(xv)
		gr, grup = theano.scan(lambda i, output, xv : T.grad(output[i], xv), 
								sequences=[T.arange(output.shape[0])], 
								non_sequences=[output,xv])
			
		self.input_set = input_set
		self.target_set = target_set
		self.batch_size = batch_size
			
		print "Compiling model..."
		self.trainNet = theano.function(inputs=[index],
					outputs=regCost,
					updates=updates,
					givens={
						x: input_set[index * batch_size:(index + 1) * batch_size],
						y: target_set[index * batch_size:(index + 1) * batch_size]})
		
		self.evalCost = theano.function(inputs=[],
					outputs=cost,
					givens={
						x: input_set,
						y: target_set})

		self.evalRegCost = theano.function(inputs=[],
					outputs=regCost,
					givens={
						x: input_set,
						y: target_set})
		
		self.evalNet = theano.function(inputs=[],
					outputs=self.n.run(x),
					givens={
						x: input_set})
		
		self.evalGrad = theano.function(inputs=[index],
					outputs=gr,
					updates=grup,
					givens={
						xv: input_set[index]})
		
		self.evalValue = theano.function(inputs=[xv],
										outputs = self.n.run(xv),
										allow_input_downcast=True)

		self.evalValueSet = theano.function(inputs=[x],
										outputs = self.n.run(x),
										allow_input_downcast=True)
		
	def train(self, nEpochs, nSamples, verbose = False):
		print "Training model with gradient descent..."
		if verbose:
			for i in xrange(nEpochs):
				for j in xrange(nSamples/self.batch_size):
					c = self.trainNet(j)
				print "Epoch", i, "cost", c
		else:
			for i in xrange(nEpochs):
				for j in xrange(nSamples/self.batch_size):
					c = self.trainNet(j)
		print "Finished training with av. error of", np.sqrt(self.evalCost())
	
	def trainDiagnostic(self, nEpochs, nSamples, prefix='net'):
		print "Training model with gradient descent..."
		reg_c = []
		true_c = []
		for i in xrange(nEpochs):
			for j in xrange(nSamples/self.batch_size):
				c = self.trainNet(j)
			reg_c.append(self.evalRegCost())
			true_c.append(self.evalCost())
		print "Finished training with av. error of", np.sqrt(self.evalCost())
		td = np.array(reg_c)
		plt.plot(td)
		tt = np.array(true_c)
		plt.plot(tt)
		plt.savefig("plots/{0}/score.pdf".format(prefix))
		plt.clf()
	
	def jitterTrain(self, nEpochs, nSamples, sigma):
		print "Training jittered model with gradient descent..."
		inputValues = self.input_set.get_value()
		for _ in xrange(nEpochs):
			jitter = inputValues + self.rng.normal(scale = sigma, size=inputValues.shape).astype("float32")
			self.input_set.set_value(jitter)
			for j in xrange(nSamples/self.batch_size):
				self.trainNet(j)
		print "Finished training with av. error of", np.sqrt(self.evalCost())
		self.input_set.set_value(inputValues)
		
	def visualizeWeights(self, prefix="net"):
		cm = plt.get_cmap("RdBu")
		for i, param in enumerate(self.n.params):
			pValues = param.get_value()
			if len(pValues.shape) == 1: #vector
				label_hist = pValues.reshape((pValues.shape[0],1))
			else:
				label_hist = pValues
			p_max = np.abs(np.max(pValues))
			p_min = np.abs(np.min(pValues))
			p_ceil = p_max if p_max > p_min else p_min
			plt.imshow(label_hist.T, interpolation='nearest', cmap = cm, vmin = -p_ceil, vmax = p_ceil)
			plt.colorbar()
			plt.title(str(pValues.shape))
			plt.savefig("plots/{2}/weights{0:02d}_p{1:02d}.pdf".format(i/2, i, prefix))
			plt.clf()
		
	def saveState(self, fname):
		data = []
		data.append(self.n.structure)
		if self.n.layers[-1].activation == T.tanh:
			a = 0
		elif self.n.layers[-1].activation == T.nnet.sigmoid:
			a = 1
		else:
			a = 2
		data.append(a)
		for l in self.n.layers:
			data.append(l.W)
			data.append(l.b)
		with file(fname, "wb") as f:
			cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
			
	def loadState(self, fname, input_set, target_set, learning_rate = 0.1, batch_size = 20, L1_reg = 0, L2_reg = 0):
		'''
		Load the neural net parameters saved in the file fname and populate them in a new
		neural net object
		'''
		with file(fname, "rb") as f:
			data = cPickle.load(f)
		
		'''
		loading in the node structure and the activation function, create net
		
		'''
		structure = data[0]
		a = data[1]
		if a == 0:
			af = T.tanh
		elif a == 1:
			af = T.nnet.sigmoid
		else: 
			af = None
		
		self.n = net.Network(self.rng, structure[1:], structure[0], activation = af)
		
		'''
		load in the parameters in the same order as they were saved
		'''
		i = 2
		for l in self.n.layers:
			l.W = data[i]
			i += 1
			l.b = data[i]
			i += 1
		
		self.n.createParameterList()
		self.buildNet(input_set, target_set, learning_rate, batch_size, L1_reg, L2_reg)