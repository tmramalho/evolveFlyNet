'''
Created on Nov 17, 2013

@author: tiago
'''

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import DataProcessor as dp
from mpl_toolkits.mplot3d import Axes3D
import time
import random
import FullModelIntegrate as model
import Network as net

class TrainModel(object):
	'''
	classdocs
	'''


	def __init__(self, ode, nnet, seed=1234):
		'''
		Constructor
		'''
		
		self.rng = np.random.RandomState(seed)
		random.seed(seed)
		self.nnet = nnet
		self.gshape = []
		input_seq = T.ftensor3("is")
		output_seq = T.ftensor3("os")
		p_init = T.fmatrix("pi")
		r_init = T.fmatrix("ri")
		k = T.fscalar("k")
		d1 = T.fscalar("d1")
		d2 = T.fscalar("d2")
		l1 = T.fscalar("l1")
		l2 = T.fscalar("l2")
		integ = model.Integrate(ode.eulerStep, dt)
		integ.buildModel(r_init, p_init, output_seq, input_seq, k, d1, d2, l1, l2)
		self.g = theano.function([input_seq, output_seq, p_init, r_init, k, d1, d2, l1, l2], 
							integ.score,
							updates=integ.updates,
							allow_input_downcast=True)
		
		self.f = theano.function([input_seq, output_seq, p_init, r_init, k, d1, d2, l1, l2], 
							(integ.r_out, integ.p_out),
							updates=integ.updates,
							allow_input_downcast=True)
		
		'''
		compute gradient w.r.t. inputs of the network
		'''
		xv = T.fvector('x')
		output = self.nnet.run(xv)
		gr, grup = theano.scan(lambda i, output, xv : T.grad(output[i], xv), 
								sequences=[T.arange(output.shape[0])], 
								non_sequences=[output,xv])
		
		self.gradient = theano.function(inputs=[xv],
					outputs=gr,
					updates=grup)
		
	def setData(self, input_sequence, output_sequence, init_state):
		self.iseq = input_sequence
		self.oseq = output_sequence
		self.pini = init_state
		self.rini = init_state
		
	def evalTrainCost(self, params):
		self.setParameterValues(params[:-5])
		if np.any(params[-5:] < 0):
			return 1000000
		k = params[-5]
		d1 = params[-4]
		d2 = params[-3]
		l1 = params[-2]
		l2 = params[-1]
		return self.g(self.iseq, self.oseq, self.pini, self.rini, k, d1, d2, l1, l2)
	
	def evalTrajectory(self, params, input_sequence=None, 
					output_sequence=None, initial_p=None, initial_r=None):
		if input_sequence is None:
			input_sequence = self.iseq
		if output_sequence is None:
			output_sequence = self.oseq
		if initial_p is None:
			initial_p = self.pini
		if initial_r is None:
			initial_r = self.rini
			
		self.setParameterValues(params[:-5])
		k = params[-5]
		d1 = params[-4]
		d2 = params[-3]
		l1 = params[-2]
		l2 = params[-1]
		return self.f(input_sequence, output_sequence, initial_p, initial_r,
					k, d1, d2, l1, l2)
		
	def getInitialParameter(self):
		x = self.getParameterValues()
		return np.concatenate((x, [1, 2, 2, 5, 5]))
		
	def trainDifferentialEvolution(self, nGenerations = 1000, absVal = 1e-5, 
									popSize = 20, F = 0.8, CR = 0.9):
		'''
		doesnt use the regularized cost
		'''
		settler = self.getInitialParameter()
		ns = settler.shape[0]
		pop = settler + self.rng.normal(scale = 0.1, 
										size=ns*popSize).reshape((popSize, ns))
		population = []
		scores = []
		for n in xrange(pop.shape[0]):
			cost = self.evalTrainCost(pop[n])
			population.append(pop[n])
			scores.append(cost)
			
		population = np.array(population)
		
		for _ in xrange(nGenerations):
			start = time.clock()
			for (j, p) in enumerate(population):
				targetIndex = self.rng.randint(0, ns)
				others = np.delete(population, j, axis=0)
				(pa, pb, pc) = random.sample(others, 3)
				trial = np.copy(p)
				for n in xrange(ns):
					r = self.rng.rand()
					if n == targetIndex or r < CR:
						trial[n] = pa[n] + F*(pb[n] - pc[n])
				cost = self.evalTrainCost(trial)
				if cost < scores[j]:
					scores[j] = cost
					population[j] = trial
			spread = np.abs(float(min(scores)) - float(max(scores)))/float(max(scores))
			print ("New gen: {0:.2f} up to {1:.2f} ({3:.2f}) in {2:.1f}".format(
					float(min(scores)), float(max(scores)), (time.clock() - start), spread))
			if spread < 0.01:
				bestIndex = scores.index(min(scores))
				self.best = population[bestIndex]
				population += self.rng.normal(scale = 0.1,
											size=ns*popSize).reshape((popSize, ns))
				population[bestIndex] = self.best
				print "EXPLODE"
			
		
		bestIndex = scores.index(min(scores))
		print population[bestIndex]
		self.best = population[bestIndex]
		np.save("results/fullBest.npy", self.best)
		
	def loadBest(self):
		self.best = np.load("results/fullBest.npy")
		
	def getParameterValues(self):
		gparams = []
		for param in self.nnet.params:
			pVal = param.get_value()
			gparams.append(pVal.flatten())
			self.gshape.append(pVal.shape)
		result = np.concatenate(gparams)
		return result
	
	def setParameterValues(self, values):
		pos = 0
		for sh, param in zip(self.gshape, self.nnet.params):
			if len(sh) == 2:
				size = sh[0]*sh[1]
			else:
				size = sh[0]
			val = values[pos:pos+size]
			param.set_value(val.reshape(sh))
			pos = pos + size
	
	def getGradient(self, t, p):
		nspace = p.shape[1]
		grad_result = []
		for n in xrange(nspace):
			inputVec = np.concatenate((self.iseq[t, n, :], p[t, n, :]))
			grad = self.gradient(inputVec)
			grad_result.append(grad)
		return np.array(grad_result)
	
	def genProbFromModel(self, scale, n_samples, nBins):
		print "Calculating histograms..."
		hist = np.zeros((self.iseq.shape[0], self.iseq.shape[1], 
						self.pini.shape[1], nBins))
		samples = []
		for _ in xrange(n_samples):
			pert_is = np.random.normal(scale=scale, size=np.prod(self.iseq.shape))
			iseq = self.iseq + pert_is.reshape(self.iseq.shape)
			pert_pi = np.random.normal(scale=scale, size=np.prod(self.pini.shape))
			pi = self.pini + pert_pi.reshape(self.pini.shape)
			_, pout = self.evalTrajectory(self.best, input_sequence=iseq, initial_p=pi)
			samples.append(pout)
		samples = np.rollaxis(np.array(samples), 0, 4)
		
		for i0 in xrange(hist.shape[0]):
			for i1 in xrange(hist.shape[1]):
				for i2 in xrange(hist.shape[2]):
					counts, _ = np.histogram(samples[i0, i1, i2], 
												nBins, (0, 1))
					hist[i0, i1, i2] = counts
		print "done"
		self.probs = hist
	
	def plotDiagnostics(self, t, dt):
		ro, po = self.evalTrajectory(self.best)
		nspace = ro.shape[1]
		ti = int(t/dt)
		cm = plt.get_cmap("RdBu")
		ap = np.linspace(0, 1, nspace)
		gs = plt.GridSpec(6, 2)
		plt.subplot(gs[0, 0])
		plt.plot(ap, self.iseq[ti, :, :])
		plt.subplot(gs[0, 1])
		plt.plot(ap, self.oseq[t, :, :])
		plt.subplot(gs[1, 0])
		plt.plot(ap, ro[ti, :, :])
		plt.subplot(gs[1, 1])
		plt.plot(ap, po[ti, :, :])
		for n in xrange(4):
			plt.subplot(gs[2+n, 0])
			gr = self.getGradient(ti, po)
			absMax = np.max(np.abs(gr[:, n]))
			plt.imshow(gr[:, n].T, interpolation='nearest', cmap = cm, 
					vmin = -absMax, vmax = absMax, aspect = 'auto')
			plt.subplot(gs[2+n, 1])
			plt.imshow(self.probs[ti, :, n, ::-1].T, interpolation= 'nearest', aspect = 'auto')
		plt.savefig("plots/diagnostic{0:02d}.pdf".format(t))
		

def visualizeGeneSpacetime(gene):
	y = np.linspace(0, 8, gene.shape[0])
	x = np.linspace(0, 100, gene.shape[1])
	x3d, y3d = np.meshgrid(x, y)
	ax = plt.figure().gca(projection='3d')
	ax.plot_surface(x3d, y3d, gene)
	ax.set_ylabel("time")
	ax.set_xlabel("space")
	ax.set_zlabel("conc")
	plt.show()

if __name__ == '__main__':
	data = dp.DataProcessor()
	dt = 0.01
	input_sequence, output_sequence, init_state = data.getInterpolatedInputOutputSequence(dt)
	#visualizeGeneSpacetime(input_sequence[:,:,0])
	n_inputs = input_sequence.shape[2]
	n_outputs = output_sequence.shape[2]
	n_spatial_steps = input_sequence.shape[1]
	n_time_steps = output_sequence.shape[0]
	rng = np.random.RandomState(1234)
	nnet = net.Network(rng, [6, n_outputs], n_inputs+n_outputs, 
						activation=theano.tensor.nnet.sigmoid)
	ode = model.ODESolver(nnet, n_spatial_steps, n_outputs, n_inputs)
	tr = TrainModel(ode, nnet)
	tr.setData(input_sequence, output_sequence, init_state)
	tr.trainDifferentialEvolution(1000)
	#tr.loadBest()
	tr.genProbFromModel(0.1, 100, 50)
	for t in xrange(n_time_steps):
		tr.plotDiagnostics(t, dt)
	
	
	
	
	