'''
Created on Sep 6, 2013

@author: tiago
'''

import DataProcessor as dp
import RecurrentNet as rn
import theano
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import argparse
import time

def loadData(data):
	print "Creating numeric arrays...",
	nsp = data.normalizedSequencesPerCell()
	
	inputs = np.rollaxis(nsp[:, :7, :3], 0, 2)
	targets = np.rollaxis(nsp[:, :, 3:7], 0, 2)#EDITEVE
	print "done"

	return inputs, targets

def getGradientInformationFromModel(model, inputs, target):
	nTimes = inputs.shape[0]
	nSamples = inputs.shape[1]
	tGrad = []
	for t in xrange(nTimes):
		sampleGrad = []
		for n in xrange(nSamples):
			inputVec = np.concatenate((inputs[t,n,:], target[t,n,:4]))
			grad = model.evalGrad(inputVec)
			sampleGrad.append(grad)
		tGrad.append(sampleGrad)
	return np.array(tGrad)

def getProbFromModel(model, inputs, target, scale, samples, nBins):
	print "Calculating histograms...",
	hist = np.zeros((inputs.shape[0], target.shape[1], target.shape[2], nBins))
	for _ in xrange(samples):
		ri = inputs + np.random.normal(scale=scale, size=np.prod(inputs.shape)).reshape(inputs.shape)
		model.input_set.set_value(ri.astype('float32'))
		rt = target + np.random.normal(scale=scale, size=np.prod(target.shape)).reshape(target.shape)
		model.target_set.set_value(rt.astype('float32'))
		out = model.evalNet()
		for i0 in xrange(hist.shape[0]):
			for i1 in xrange(hist.shape[1]):
				for i2 in xrange(hist.shape[2]):
					index = out[i0, i1, i2] * nBins
					hist[i0, i1, i2, index] += 1
	print "done"
	return hist
	
def plotResults(data, model):
	inputs, target = loadData(data)
	ap = np.linspace(0, 1, 100)
	nTimes = inputs.shape[0]
	nSamples = inputs.shape[1]
	nTargets = target.shape[2]
	result = model.evalNet()
	gradients = getGradientInformationFromModel(model, inputs, target)
	cm = plt.get_cmap("RdBu")
	genes = data.getGeneNames()
	probs = getProbFromModel(model, inputs, target, 0.05, 5000, 50)
	print "Plotting...",
	for t in xrange(nTimes):
		for i in xrange(nSamples/100):
			plt.figure(figsize=(50,10))
			for k in xrange(nTargets):
				plt.subplot(4, nTargets, k+1)
				plt.plot(ap, inputs[t, i*100:(i+1)*100])
				plt.plot(ap, target[t, i*100:(i+1)*100, :4])
				plt.title("Result at time class "+str(t) + " for gene " + genes[k+3])
				plt.subplot(4, nTargets, k+1+nTargets)
				plt.plot(ap, target[t+1, i*100:(i+1)*100,k], label='real')
				plt.plot(ap, result[t, i*100:(i+1)*100,k], label='inferred')
				plt.legend()
				plt.subplot(4, nTargets, k+1+nTargets*2)
				gr = gradients[t]
				absMax = np.max(np.abs(gr[i*100:(i+1)*100, k]))
				plt.imshow(gr[i*100:(i+1)*100, k].T, interpolation='nearest', cmap = cm, vmin = -absMax, vmax = absMax, aspect = 'auto')
				plt.gca().set_yticks([0,1,2,3,4,5,6])
				plt.gca().set_yticklabels(genes[:7])
				plt.subplot(4, nTargets, k+1+nTargets*3)
				plt.imshow(probs[t, i*100:(i+1)*100, k, ::-1].T, interpolation= 'nearest', aspect = 'auto')
			plt.savefig("plots/recurrentNetwork"+str(i)+"_t"+str(t)+".pdf")
			plt.clf()
	
	cm = plt.get_cmap("RdBu")
	for i, param in enumerate(model.n.params):
		pValues = param.get_value()
		if len(pValues.shape) == 1: #vector
			label_hist = pValues.reshape((pValues.shape[0],1))
		else:
			label_hist = pValues
		plt.imshow(label_hist.T, interpolation='nearest', cmap = cm, vmin = -6, vmax = 6)
		plt.colorbar()
		plt.title(str(pValues.shape))
		plt.savefig("plots/recurrentNetWeightsLayer"+str(i/2)+"_"+str(i%2)+".pdf")
		plt.clf()
	
	print "done"

def populateModel(args, data, load, string, sigma = None):
	inputs, target = loadData(data)
	input_set = theano.shared(inputs.astype("float32"))
	target_set = theano.shared(target.astype("float32"))
	model = rn.RecurrentNet(int(time.time()), sigma)
	if load:
		model.loadState(string, input_set, target_set)
	else:
		iDim = inputs.shape[2] + target.shape[2] #EDITEVE do not consider eve
		model.createNewNet(iDim, target.shape[2], input_set, target_set, L1_reg = 0.00001, nHidden = args.nh)
	return model

def trainRecurrentNetworkDE(args, data, load = False):
	model = populateModel(args, data, load, "data/nets/recurrent_h"+str(args.nh)+"Net.nn", args.sigma)
	model.trainDifferentialEvolution(nGenerations = args.n, popSize = 40, F = 0.6)
	model.saveState("data/nets/recurrent_h"+str(args.nh)+"Net.nn")
	return model
	
def trainRecurrentNetworkSGD(args, data, load = False):
	model = populateModel(args, data, load, "data/nets/recurrent_h"+str(args.nh)+"Net.nn", args.sigma)
	model.train(args.n, verbose = True)
	model.saveState("data/nets/recurrent_h"+str(args.nh)+"Net.nn")
	return model

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train a network')
	parser.add_argument('--ne', dest='n',
					type=int, default=1000,
					help="number of training epochs")
	parser.add_argument('--nh', dest='nh',
					type=int, default=10,
					help="number of hidden units")
	parser.add_argument('--s', dest='sigma',
					type=float, default=None,
					help="noise to apply to units")
	parser.add_argument('--scan', action='store_true', help="train all genes")
	parser.add_argument('--f', dest='folder',
					type=str, default='/Users/tiago/Dropbox/workspace/evolveFlyNet/',
					help="folder")
	
	args = parser.parse_args()
	data = dp.DataProcessor(args.folder)
	
	model = trainRecurrentNetworkDE(args, data)
	#model = trainRecurrentNetworkSGD(args, data, load=True)
	#model = populateModel(args, data, True, "data/nets/recurrent_h"+str(args.nh)+"Net.nn")
	plotResults(data, model)