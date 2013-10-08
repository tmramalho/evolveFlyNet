'''
Created on Sep 7, 2013

@author: tiago
'''

import DataProcessor as dp
import SGDOptimizer as so
import Network as net
import Integrate as ig
import theano
import theano.sandbox.rng_mrg
import theano.tensor as T
import numpy as np
import scipy.interpolate as ip
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cPickle
import random
import time

class RecurrentNet(object):
	'''
	classdocs
	'''
	
	def __init__(self, seed=1234, sigma= None):
		self.rng = np.random.RandomState(seed)
		random.seed(seed)
		#self.trng = T.shared_randomstreams.RandomStreams(seed)
		self.trng = theano.sandbox.rng_mrg.MRG_RandomStreams(seed)
		self.sigma = sigma
		
	def createNewNet(self, iDim, oDim, input_set, target_set, 
				activation = T.nnet.sigmoid, learning_rate = 0.1, batch_size = 20, L1_reg = 0, L2_reg = 0,
				nHidden = 10, nNets = 5):
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
	
	def rnnStep(self, u, x):
		'''
		Concatenate the input sequence (maternal morphogens)
		With the current state (gap genes)
		Remove the last element of the state because it's not a gap gene
		x[:, :-1]
		#EDITEVE only considering gap genes for now
		'''
		netInput = T.concatenate([u, x[:, :]], axis = 1)
		result = self.n.run(netInput)
		return T.cast(result, 'float32')
	
	def rnnStochasticStep(self, u, x):
		'''
		Concatenate the input sequence (maternal morphogens)
		With the current state (gap genes)
		Remove the last element of the state because it's not a gap gene
		x[:, :-1]
		#EDITEVE only considering gap genes for now
		'''
		netInput = T.concatenate([u, x[:, :]], axis = 1)
		#r = self.trng.normal(size = netInput.shape, std = self.sigma)
		r = self.trng.uniform(size=netInput.shape, high = self.sigma)
		result = self.n.run(netInput + r)
		return T.cast(result, 'float32')
	
	def buildNet(self, input_set, target_set, learning_rate, batch_size, L1_reg, L2_reg):
		'''
		Creates the cost function and gradient expressions, compiles the model
		
		Keyword arguments:
		input_set -- theano shared variable containing input data [#times, #samples, #inputs]
		target_set -- theano shared variable containing target data [#times, #samples, #outputs]
		learning_rate -- multiplicative constant for the gradient
		l1_reg -- L1 regularization penalty
		l2_reg -- L2 regularization penalty
		batch_size -- number of samples to be evaluated simultaneously
		'''
		index = T.lscalar("ix")
		u = T.ftensor3('u')
		x = T.fmatrix('x')
		y = T.ftensor3('y')
		
		(self.coutStoch, scanUpdatesStoch) = theano.scan(fn = self.rnnStochasticStep,
												outputs_info = [x],
												sequences = [u],
												n_steps = u.shape[0])

		(self.cout, scanUpdates) = theano.scan(fn = self.rnnStep,
												outputs_info = [x],
												sequences = [u],
												n_steps = u.shape[0])
		
		"""Least square difference"""
		dist = y - self.cout
		cost = T.mean(dist ** 2)
		regCost = cost + L1_reg * self.n.L1 + L2_reg * self.n.L2_sqr
		regCostStoch = cost + L1_reg * self.n.L1 + L2_reg * self.n.L2_sqr
		
		gparams = []
		for param in self.n.params:
			gparam = T.grad(cost, param)
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
			
		print "Compiling model...",
		self.trainNet = theano.function(inputs=[index],
					outputs=regCost,
					updates=updates + scanUpdates,
					givens={
						u: input_set[:, index * batch_size:(index + 1) * batch_size, :],
						x: target_set[0, index * batch_size:(index + 1) * batch_size, :],
						y: target_set[1:, index * batch_size:(index + 1) * batch_size, :]})
		
		if self.sigma is not None:
			self.evalTrainCost = theano.function(inputs=[],
					outputs=regCostStoch,
					updates = scanUpdatesStoch,
					givens={
						u: input_set[:, :, :],
						x: target_set[0, :, :],
						y: target_set[1:, :, :]})
		else:
			self.evalTrainCost = theano.function(inputs=[],
					outputs=regCost,
					updates = scanUpdates,
					givens={
						u: input_set[:, :, :],
						x: target_set[0, :, :],
						y: target_set[1:, :, :]})
		
		self.evalCost = theano.function(inputs=[],
					outputs=cost,
					updates = scanUpdates,
					givens={
						u: input_set[:, :, :],
						x: target_set[0, :, :],
						y: target_set[1:, :, :]})
		
		self.evalNet = theano.function(inputs=[],
					outputs=self.cout,
					updates = scanUpdates,
					givens={
						u: input_set[:, :, :],
						x: target_set[0, :, :]})
		
		
		self.evalGrad = theano.function(inputs=[xv],
					outputs=gr,
					updates=grup)
		print "done"
		
	def getParameterValues(self):
		gparams = []
		self.gshape = []
		for param in self.n.params:
			pVal = param.get_value()
			gparams.append(pVal.flatten())
			self.gshape.append(pVal.shape)
		result = np.concatenate(gparams)
		return result
	
	def setParameterValues(self, values):
		pos = 0
		for sh, param in zip(self.gshape, self.n.params):
			if len(sh) == 2:
				size = sh[0]*sh[1]
			else:
				size = sh[0]
			val = values[pos:pos+size]
			param.set_value(val.reshape(sh))
			pos = pos + size
			
	def trainDifferentialEvolution(self, nGenerations = 1000, absVal = 1e-5, popSize = 20, F = 0.8, CR = 0.9):
		'''
		doesnt use the regularized cost
		'''
		settler = self.getParameterValues()
		ns = settler.shape[0]
		pop = settler + self.rng.normal(scale = 0.1, size=ns*popSize).reshape((popSize, ns))
		population = []
		scores = []
		for n in xrange(pop.shape[0]):
			self.setParameterValues(pop[n])
			cost = self.evalTrainCost()
			population.append(pop[n])
			scores.append(cost)
		
		for _ in xrange(nGenerations):
			start = time.clock()
			for (j, p) in enumerate(population):
				targetIndex = self.rng.randint(0, ns)
				others = population[:j] + population[(j + 1):]
				(pa, pb, pc) = random.sample(others, 3)
				trial = np.copy(p)
				for n in xrange(ns):
					r = self.rng.rand()
					if n == targetIndex or r < CR:
						trial[n] = pa[n] + F*(pb[n] - pc[n])
				self.setParameterValues(trial)
				cost = self.evalTrainCost()
				if cost < scores[j]:
					scores[j] = cost
					population[j] = trial
			print "New generation with score", min(scores), "up to", max(scores), "in", (time.clock() - start)*1000
		
		bestIndex = scores.index(min(scores))
		self.setParameterValues(population[bestIndex])
		
	def train(self, nEpochs, verbose = False):
		print "Training model with gradient descent..."
		nSamples = self.input_set.get_value().shape[1]
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
	
	def jitterTrain(self, nEpochs, sigma):
		print "Training jittered model with gradient descent..."
		inputValues = self.input_set.get_value()
		nSamples = inputValues.shape[1]
		for _ in xrange(nEpochs):
			jitter = inputValues + self.rng.normal(scale = sigma, size=inputValues.shape).astype("float32")
			self.input_set.set_value(jitter)
			for j in xrange(nSamples/self.batch_size):
				self.trainNet(j)
		print "Finished training with av. error of", np.sqrt(self.evalCost())
		self.input_set.set_value(inputValues)
		
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