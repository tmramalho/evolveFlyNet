'''
Created on Jan 24, 2014

@author: tiago
'''

import numpy as np
import theano
import time
import theano.tensor as T
import matplotlib.pyplot as plt
import abstractNetwork as net
import scipy.interpolate as ip
from mpl_toolkits.mplot3d import Axes3D
import DataProcessor as dp
import random

class AbstractModel(object):
	'''
	classdocs
	'''
	

	def __init__(self, data, seed=1234, nh=10):
		'''
		Constructor
		'''
		in_seq, out_seq, _ = data.getInterpolatedInputOutputSequence(0.1, 0)
		self.final_y = self.smoothAndNormalize(out_seq[6], 0.01)
		self.input_seq = self.smoothAndNormalize(in_seq[0], 0.01)
		self.out_dim = self.final_y.shape[1]
		self.in_dim = self.input_seq.shape[1]
		nid = self.in_dim + 2*self.out_dim
		self.rng = np.random.RandomState(seed)
		self.net = net.Network(self.rng, [nh, self.out_dim], nid)
		self.initial_state = theano.shared(np.ones(self.final_y.shape, 'float32'))
		self.do = self.create_diffusion_op(self.final_y.shape[0])
		
	def combinedRK4Step(self, c, inp, dt):
		cd1 = T.dot(self.do, c)
		z1 = T.concatenate([inp, c, cd1], axis=1)
		k1 = self.net.run(z1)
		cd2 = T.dot(self.do, c+dt*k1/2)
		z2 = T.concatenate([inp, c+dt*k1/2, cd2], axis=1)
		k2 = self.net.run(z2)
		cd3 = T.dot(self.do, c+dt*k2/2)
		z3 = T.concatenate([inp, c+dt*k2/2, cd3], axis=1)
		k3 = self.net.run(z3)
		cd4 = T.dot(self.do, c+dt*k3)
		z4 = T.concatenate([inp, c+dt*k3, cd4], axis=1)
		k4 = self.net.run(z4)
		return T.cast(c + dt*(k1 + 2*k2 + 2*k3 + k4)/6, "float32")
	
	def setup_deterministic_solver(self, dt=0.01, L1_reg=0, L2_reg=0):
		total_steps = T.cast(1/dt, 'int32')
		(self.cout, self.updates) = theano.scan(fn = self.combinedRK4Step,
									outputs_info = [self.initial_state],
									non_sequences = [self.input_seq, dt],
									n_steps = total_steps)
		
		cost = T.sum(T.pow(self.cout[-1] - self.final_y,2)) + L1_reg * self.net.L1 + L2_reg * self.net.L2_sqr
		
		self.f = theano.function([], cost, updates=self.updates)
		self.fe = theano.function([], self.cout[-1], updates=self.updates)
	
	def getParameterValues(self):
		gparams = []
		self.gshape = []
		for param in self.net.params:
			pVal = param.get_value()
			gparams.append(pVal.flatten())
			self.gshape.append(pVal.shape)
		result = np.concatenate(gparams)
		return result
	
	def setParameterValues(self, values):
		pos = 0
		for sh, param in zip(self.gshape, self.net.params):
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
			cost = self.f()
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
				cost = self.f()
				if cost < scores[j]:
					scores[j] = cost
					population[j] = trial
			print "New generation with score", min(scores), "up to", max(scores), "in", (time.clock() - start)*1000
		
		bestIndex = scores.index(min(scores))
		self.setParameterValues(population[bestIndex])
	
	def create_diffusion_op(self, space_steps):
		self.space_steps = space_steps
		diffusion_matrix = -2*np.diag(np.ones(self.space_steps))
		diagonal = np.diag_indices(self.space_steps)
		end_pos = self.space_steps - 1
		diffusion_matrix[((diagonal[0] + 1) % self.space_steps, diagonal[1])] = 1
		diffusion_matrix[((diagonal[0] - 1) % self.space_steps, diagonal[1])] = 1
		diffusion_matrix[0, 0] = -1
		diffusion_matrix[end_pos, 0] = 0
		diffusion_matrix[end_pos, end_pos] = -1
		diffusion_matrix[0, end_pos] = 0
		return diffusion_matrix
	
	def smoothAndNormalize(self, points, sm=None):
		l = np.linspace(0, 1, points.shape[0])
		spoints = np.copy(points)
		for i in xrange(points.shape[1]):
			sp = ip.UnivariateSpline(l, points[:, i], s=sm)
			spoints[:, i] = np.clip(sp(l)/np.max(sp(l)), 0, 1)
		return spoints
		
		
if __name__ == '__main__':
	data = dp.DataProcessor()
	rn = AbstractModel(data)
	rn.setup_deterministic_solver(0.01, 0, 0)
	rn.trainDifferentialEvolution()
	fy = rn.fe()
	plt.plot(fy)
	plt.show()