'''
Created on Aug 19, 2013

@author: tiago
'''

import numpy as np
import theano
import theano.tensor as T
import Network as net

class ODESolver(object):
	"""
	Keeps all methods for integrating a single step of an ode
	"""
	def __init__(self, network):
		"""
		Stores the network model which gives the dynamics for the system
			
		Keyword arguments:
		network -- a Network object which will be evaluated at each step
		"""
		self.n = network

	def eulerStep(self, inp, c, dt):
		fn = self.n.run(inp)
		return T.cast(c + dt*fn, "float32")

	def combinedEulerStep(self, inp, c, dt):
		z = T.concatenate([inp, c])
		fn = self.n.run(z)
		return T.cast(c + dt*fn, "float32")
	
	def combinedRK4Step(self, inp, c, dt):
		z1 = T.concatenate([inp, c])
		k1 = self.n.run(z1)
		z2 = T.concatenate([inp, c+dt*k1/2])
		k2 = self.n.run(z2)
		z3 = T.concatenate([inp, c+dt*k2/2])
		k3 = self.n.run(z3)
		z4 = T.concatenate([inp, c+dt*k3])
		k4 = self.n.run(z4)
		return T.cast(c + dt*(k1 + 2*k2 + 2*k3 + k4)/6, "float32")

class Integrate(object):
	def __init__(self, inputs, outputs, c0, odeSolver, dt=0.01):
		"""
		Builds the expression containing the integration loop; and
		the expression comparing the result of the integration loop
		with the desired output with a least squares measure
			
		Keyword arguments:
		inputs -- a sequence with the control inputs for the network
		outputs -- a sequence with the desired output for the network
					(one measurement per dimensionless time unit)
		c0 -- initial system state
		odeSolver -- method which performs the integration at each time step
		"""
		stepsPerUnit = T.cast(1/dt,'int32')
		numUnits = outputs.shape[0]
		total_steps = T.cast(numUnits*stepsPerUnit, 'int32')
		(self.cout, self.updates) = theano.scan(fn = odeSolver,
									outputs_info = [c0],
									sequences = [inputs],
									non_sequences = [dt],
									n_steps = total_steps)
		
		"""Sim results only for each time unit"""
		cUnits = self.cout[stepsPerUnit-1::stepsPerUnit]
		
		"""Least square difference"""
		dist = outputs - cUnits
		self.score = (dist ** 2).sum()

def constantInputTest():
	rng = np.random.RandomState(1234)
	n = net.Network(rng, [8,2], 5)
	o = ODESolver(n)
	iSeq = theano.shared(rng.rand(300,5))
	oSeq = theano.shared(np.array([[1,1],[0.2,0.3],[0.4,0.4]], dtype='float32'))
	c0 = theano.shared(np.array([0.1,0.1], dtype='float32'))
	integ = Integrate(iSeq, oSeq, c0, o.eulerStep)
	#f = theano.function([], integ.cout, updates=integ.updates)
	#print "Running scan:", f()
	g = theano.function([], integ.score, updates=integ.updates)
	print "Difference:", g()
	c0.set_value(np.array([0.6,0.6], dtype='float32'))
	print "Difference:", g()
	iSeq.set_value(rng.rand(300,5))
	print "Difference:", g()

def variableInputTest():
	rng = np.random.RandomState(1234)
	n = net.Network(rng, [8,2], 5)
	o = ODESolver(n)
	iSeq = theano.shared(rng.rand(300,3))
	oSeq = theano.shared(np.array([[1,1],[0.2,0.3],[0.4,0.4]], dtype='float32'))
	c0 = theano.shared(np.array([0.1,0.1], dtype='float32'))
	integ = Integrate(iSeq, oSeq, c0, o.combinedEulerStep)
	#f = theano.function([], integ.cout, updates=integ.updates)
	#print "Running scan:", f()
	g = theano.function([], integ.score, updates=integ.updates)
	print "Difference:", g()
	c0.set_value(np.array([0.6,0.6], dtype='float32'))
	print "Difference:", g()
	iSeq.set_value(rng.rand(300,3))
	print "Difference:", g()
	theano.printing.pydotprint(g, "scan.png")


def symbolicTest():
	rng = np.random.RandomState(1234)
	n = net.Network(rng, [8,2], 5)
	o = ODESolver(n)
	iSeq = theano.shared(np.array(rng.rand(300,5), dtype='float32'))
	oSeq = theano.shared(np.array([[1,1],[0.2,0.3],[0.4,0.4]], dtype='float32'))
	c0 = theano.shared(np.array([0.1,0.1], dtype='float32'))
	inputSequence = T.fmatrix("is")
	outputSequence = T.fmatrix("os")
	initialState = T.fvector("c_i")
	integ = Integrate(inputSequence, outputSequence, initialState, o.eulerStep)
	#f = theano.function([], integ.cout, updates=integ.updates)
	#print "Running scan:", f()
	g = theano.function([], integ.score, updates=integ.updates,
					givens={inputSequence: iSeq,
						outputSequence: oSeq,
						initialState: c0})
	print "Difference:", g()

if __name__ == '__main__':
	symbolicTest()