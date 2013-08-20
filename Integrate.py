'''
Created on Aug 19, 2013

@author: tiago
'''

import numpy as np
import theano
import theano.tensor as T
import Network as net

def eulerStep(inp, c, dt):
	return T.cast(c + dt*n.run(inp), "float32")

class Integrate(object):
	def __init__(self, inputs, outputs, c0, dt=0.01):
		stepsPerUnit = T.cast(1/dt,'int32')
		numUnits = outputs.shape[0]
		total_steps = T.cast(numUnits*stepsPerUnit, 'int32')
		(self.cout, self.updates) = theano.scan(fn = eulerStep,
									outputs_info = [c0],
									sequences = [inputs],
									non_sequences = [dt],
									n_steps = total_steps)
		cUnits = self.cout[stepsPerUnit-1::stepsPerUnit]
		dist = outputs - cUnits
		self.score = (dist ** 2).sum()

if __name__ == '__main__':
	rng = np.random.RandomState(1234)
	x = theano.shared(np.ones(5))
	n = net.Network(rng, [8,2], 5)
	iSeq = theano.shared(np.random.rand(300,5))
	oSeq = theano.shared(np.array([[1,1],[0.2,0.3],[0.4,0.4]], dtype='float32'))
	c0 = theano.shared(np.array([0.1,0.1], dtype='float32'))
	integ = Integrate(iSeq, oSeq, c0)
	f = theano.function([], integ.cout, updates=integ.updates)
	print "Running scan:", f()
	g = theano.function([], integ.score, updates=integ.updates)
	print "Difference:", g()
	c0.set_value(np.array([0.6,0.6], dtype='float32'))
	print "Difference:", g()
	iSeq.set_value(np.random.rand(300,5))
	print "Difference:", g()