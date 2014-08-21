'''
Created on Aug 19, 2013

@author: tiago
'''

import numpy as np
import theano
import theano.tensor as T
import Network as net
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ODESolver(object):
	"""
	Keeps all methods for integrating a single step of an ode
	"""
	def __init__(self, network, space_steps, n_species, n_inputs = 0):
		"""
		Stores the network model which gives the dynamics for the system
			
		Keyword arguments:
		network -- a Network object which will be evaluated at each step
		space_steps -- the number of cells in the spatial discretization
		n_species -- the number of chemical species to be simulated
		"""
		self.net = network
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
		self.diffusion_matrix = diffusion_matrix
		self.n_species = n_species
		self.result = T.fmatrix("res")
		self.n_inputs = 0
		self.pa = theano.shared(np.zeros((space_steps, n_species+n_inputs), dtype='float32'))
		
	def dynamic_system(self, inp, r, p, k, d1, d2, l1, l2):
		'''
		Calculates the following dynamic system for the input:
		dr/dt = f(p) + D1*grad^2.r - l1*r
		dp/dt = k*r  + D2*grad^2.p - l2*p
		
		Keyword arguments:
		inp -- external protein concentrations
		r, p -- protein and rna concentration
		k -- p production rate / decay length
		d1, d2 -- diffusion rates / decay length
		
		Notes: inp and p must be merged: i use set_subtensor,
		faster than concatenate because can do it inplace
		'''
		T.set_subtensor(self.pa[:, :self.n_inputs], inp, inplace=True)
		T.set_subtensor(self.pa[:, self.n_inputs:], p, inplace=True)
		rn = (self.net.run(self.pa) + d1*theano.dot(self.diffusion_matrix, r) - l1*r)
		pn = (k*r + d2*theano.dot(self.diffusion_matrix, p) - l2*p)
		return rn,pn
		
	def eulerStep(self, inp, r, p, dt, k, d1, d2, l1, l2):
		frn, fpn = self.dynamic_system(inp, r, p, k, d1, d2, l1, l2)
		return T.cast(r + dt*frn, "float32"), T.cast(p + dt*fpn, "float32")
		
	def combinedRK4Step(self, inp, c, dt):
		z1 = T.concatenate([inp, c], axis=1)
		k1 = self.dynamic_system(z1)
		z2 = T.concatenate([inp, c+dt*k1/2], axis=1)
		k2 = self.dynamic_system(z2)
		z3 = T.concatenate([inp, c+dt*k2/2], axis=1)
		k3 = self.dynamic_system(z3)
		z4 = T.concatenate([inp, c+dt*k3], axis=1)
		k4 = self.dynamic_system(z4)
		return T.cast(c + dt*(k1 + 2*k2 + 2*k3 + k4)/6, "float32")
		
class Integrate(object):
	def __init__(self, ode_solver, dt=0.01):
		"""
		Stores the important parameters
		
		Keyword arguments:
		ode_solver -- method which performs the integration at each time step
		dt -- time step for integration
		"""
		self.ode_solver = ode_solver
		self.dt = dt
		
	def buildModel(self, rin, pin, pout, inputs, k, d1, d2, l1, l2):
		"""
		Builds the expression containing the integration loop; and
		the expression comparing the result of the integration loop
		with the desired output with a least squares measure
			
		Keyword arguments:
		inputs -- a sequence with the control inputs for the network
		outputs -- a sequence with the desired output for the network
					(one measurement per dimensionless time unit)
		c0 -- initial system state
		"""
		stepsPerUnit = T.cast(1/self.dt,'int32')
		numUnits = pout.shape[0]
		total_steps = T.cast(numUnits*stepsPerUnit, 'int32')
		([self.r_out, self.p_out], self.updates) = theano.scan(fn = self.ode_solver,
									outputs_info = [rin, pin],
									sequences = [inputs],
									non_sequences = [self.dt, k, d1, d2, l1, l2],
									n_steps = total_steps)
		
		"""Sim results only for each time unit"""
		self.cUnits = self.p_out[stepsPerUnit-1::stepsPerUnit]
		
		"""Least square difference"""
		dist = pout - self.cUnits
		self.score = (dist ** 2).sum()
		self.mean = T.mean(dist ** 2)

def symbolicTest():
	rng = np.random.RandomState(1234)
	n = net.Network(rng, [8,2], 7, activation=theano.tensor.nnet.sigmoid)
	n_inputs = 5
	n_outputs = 2
	n_spatial_steps = 100
	n_time_steps = 3
	dt = 0.01
	o = ODESolver(n, n_spatial_steps, n_outputs, n_inputs)
	iSeq = theano.shared(np.array(rng.rand(n_time_steps/dt, n_spatial_steps, n_inputs), dtype='float32'))
	oSeq = theano.shared(np.array(rng.rand(n_time_steps, n_spatial_steps, n_outputs), dtype='float32'))
	init = np.zeros((n_spatial_steps, n_outputs), dtype='float32')
	init[50, 0] = 10
	init[20, 1] = 1
	p0 = theano.shared(init)
	r0 = theano.shared(init)
	input_seq = T.ftensor3("is")
	output_seq = T.ftensor3("os")
	p_init = T.fmatrix("pi")
	r_init = T.fmatrix("ri")
	k = T.fscalar("k")
	d1 = T.fscalar("d1")
	d2 = T.fscalar("d2")
	l1 = T.fscalar("l1")
	l2 = T.fscalar("l2")
	integ = Integrate(o.eulerStep, dt)
	integ.buildModel(r_init, p_init, output_seq, input_seq, k, d1, d2, l1, l2)
	print "Compiling model"
	g = theano.function([k, d1, d2, l1, l2], (integ.score, integ.p_out, integ.r_out),
					updates=integ.updates,
					givens={input_seq: iSeq,
						output_seq: oSeq,
						p_init: p0,
						r_init: r0},
					allow_input_downcast=True)
	print "Finished compilation"
	diff, p_out, r_out = g(3, 30, 30, 10, 10)
	print "Difference", diff
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x = np.linspace(0, 1, 100)
	y = np.linspace(0, 3, 300)
	xt, yt = np.meshgrid(x, y)
	ax.plot_surface(xt, yt, p_out[:, :, 1], alpha = 0.5)
	ax.plot_surface(xt, yt, r_out[:, :, 1], color='r', alpha = 0.5)
	plt.show()

if __name__ == '__main__':
	symbolicTest()