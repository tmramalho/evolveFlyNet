'''
Created on Aug 23, 2013

@author: tiago
'''

import DataProcessor as dp
import SGDOptimizer as so
import Network as net
import Integrate as ig
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

def plotData(data):
	names = data.getGeneNames()
	numbers = data.normData
	x = np.linspace(0, 1, 100)
	for g in xrange(len(names)):
		for i in xrange(8):
			plt.plot(x, numbers[i,g,0,:])
		plt.title(names[g])
		plt.savefig("plots/gene_"+names[g]+".pdf")
		plt.clf()
			
	
if __name__ == '__main__':
	data = dp.DataProcessor()
	plotData(data)
	dt = 0.01
	nOutputGenes = 5
	nInputGenes = 3
	nTotalGenes = nInputGenes + nOutputGenes
	rng = np.random.RandomState(1234)
	nsp = data.normalizedSequencesPerCell()
	n = net.Network(rng, [8,2], 5)
	o = ig.ODESolver(n)
	integ = ig.Integrate(o.combinedEulerStep, dt)
	
	iSamples = np.array(nsp[:,:,:3], dtype='float32')
	'''Select last 5 genes as target values'''
	oSamples = np.array(nsp[:,1:,3:], dtype='float32')
	'''Select first time point of last 5 genes as initial condition'''
	c0Samples = np.array(nsp[:,1,3:], dtype='float32')
	
	stepsPerUnit = T.cast(1/dt,'int32')
	numUnits = oSamples.shape[1]
	total_steps = T.cast(numUnits*stepsPerUnit, 'int32')
	
	#interpolate the isamples
	
	'''
	iSeq = np.array(rng.rand(300,3)
	oSeq = np.array([[1,1],[0.2,0.3],[0.4,0.4]], dtype='float32')
	c0 = theano.shared(np.array([0.1,0.1], dtype='float32'))
	iSamples = theano.shared(np.tile(iSeq, (50,1,1)))
	oSamples = theano.shared(np.tile(oSeq, (50,1,1)))
	n = net.Network(rng, [8,2], 5)
	o = ig.ODESolver(n)
	sgd = so.SGDOptimizer(n, o.combinedEulerStep, iSamples, oSamples, c0)
	print "Dry run!"
	sgd.trainTestModel()'''