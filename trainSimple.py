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
import scipy.interpolate as ip
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
		for i in xrange(8):
			plt.plot(x, numbers[i,g,1,:])
		plt.title(names[g])
		plt.savefig("plots/gene_"+names[g]+"_dtll.pdf")
		plt.clf()
			
	
def interpolateSingleGene(data, totalSteps, plot=False):
	nc = data.shape[0] #num cellss
	nm = data.shape[1] #num measurements
	x = np.linspace(0, 100, nc) #ap axis position
	y = np.linspace(0, totalSteps, nm)
	spline = ip.RectBivariateSpline(x, y, data)
	yFull = np.linspace(0, totalSteps, totalSteps)
	result = np.array(spline(x,yFull))
	x3d, y3d = np.meshgrid(x, yFull)
	if(plot):
		ax = plt.figure().gca(projection='3d')
		ax.plot_surface(x3d, y3d, result.T)
		plt.show()
	return result

def plotResult(res, nog):
	x = np.linspace(0,1,100)
	n1 = res.shape[2]
	n2 = res.shape[1]
	for i in range(n1):
		for j in range(n2):
			plt.plot(x, res[:100,j,i])
		plt.savefig("result_"+str(i)+".pdf")
		plt.clf()
		for j in range(n2):
			plt.plot(x, res[100:,j,i])
		plt.savefig("result_"+str(i)+"_dtll.pdf")
		plt.clf()
	
	
def loadData(folder):
	data = dp.DataProcessor()
	#plotData(data)
	nsp = data.normalizedSequencesPerCell()
	'''Select last 5 genes as target values'''
	oSamples = np.array(nsp[:,1:,3:], dtype='float32')
	'''Select first time point of last 5 genes as initial condition'''
	c0Samples = np.array(nsp[:,0,3:], dtype='float32')
	
	stepsPerUnit = 1/dt
	numUnits = oSamples.shape[1]
	totalSteps = numUnits*stepsPerUnit
	
	'''Create input genes array'''
	interpGenes = []
	for g in xrange(3):
		interpGenes.append(interpolateSingleGene(nsp[:,:,g], totalSteps))
	
	iSamples = np.array(interpGenes, dtype='float32')
	iSamples = np.rollaxis(iSamples, 0, 3).clip(min = 0)
	
	np.save(folder+"data/simpleInputSequence.npy", iSamples)
	np.save(folder+"data/simpleOutputSequence.npy", oSamples)
	np.save(folder+"data/simpleStartSequence.npy", c0Samples)
	
	return iSamples, oSamples, c0Samples
	
if __name__ == '__main__':
	dt = 0.01
	nOutputGenes = 5
	nInputGenes = 3
	nTotalGenes = nInputGenes + nOutputGenes
	rng = np.random.RandomState()
	n = net.Network(rng, [10, nOutputGenes], nTotalGenes)
	o = ig.ODESolver(n)
	integ = ig.Integrate(o.combinedEulerStep, dt)
	folder = '/Users/tiago/Dropbox/workspace/evolveFlyNet/'
	
	try:
		iSamples = np.load(folder+"data/simpleInputSequence.npy")
		oSamples = np.load(folder+"data/simpleOutputSequence.npy")
		c0Samples = np.load(folder+"data/simpleStartSequence.npy")
	except IOError:
		iSamples, oSamples, c0Samples = loadData(folder)
	
	inputSeq = theano.shared(iSamples)
	outputSeq = theano.shared(oSamples)
	starting = theano.shared(c0Samples)
	sgd = so.SGDOptimizer(n, integ, inputSeq, outputSeq, starting, learning_rate = 0.15)
	print "Dry run!"
	for j in range(0,50):
		av = 0
		for index in range(0,200):
			score = sgd.model(index)
			av += score
			print j, index % 100, score
		print "avScore", av/200
	print "whopper"
	outputs = []
	for index in range(0,200):
		op = sgd.eval(index)
		outputs.append(op)
	result = np.array(outputs)
	plotResult(result, nOutputGenes)
		