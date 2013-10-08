'''
Created on Sep 6, 2013

@author: tiago
'''

import DataProcessor as dp
import SimpleNet as sn
import theano
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import cPickle

def plotData(data, names):
	x = np.linspace(0, 1, 100)
	for g in xrange(data.shape[2]):
		for i in xrange(8):
			plt.plot(x, data[:100,i,g])
		plt.title(names[g])
		plt.savefig("plots/gene_"+names[g]+".pdf")
		plt.clf()
		for i in xrange(8):
			plt.plot(x, data[100:,i,g])
		plt.title(names[g])
		plt.savefig("plots/gene_"+names[g]+"_dtll.pdf")
		plt.clf()

def loadData(data):
	print "Creating numeric arrays..."
	nsp = data.normalizedSequencesPerCell()
	#plotData(nsp, geneNames)
	
	targets = nsp[:, :, 3:]
	control = nsp[:, :, :3]

	return control, targets

def plotResults(real, inferred1, inferred2, grad, hist, genes, tstep):
	ap = np.linspace(0, 1, 100)
	nSamples = real.shape[0]
	nGenes = real.shape[1]
	cm = plt.get_cmap("RdBu")
	gs = gridspec.GridSpec(3, 2, width_ratios=[15,1])
	gs.update(wspace=0.02)
	for i in xrange(nSamples/100):
		for j in xrange(nGenes):
			plt.figure(figsize = (10, 15))
			g = grad[j]
			plt.subplot(gs[0, 0])
			plt.plot(ap, real[i*100:(i+1)*100,j], label='real')
			plt.plot(ap, inferred1[i*100:(i+1)*100,j], label='dyn_inf')
			plt.plot(ap, inferred2[i*100:(i+1)*100,j], label='mea_inf')
			plt.title(genes[j+3]+" at time class "+str(tstep) + " for type " + str(i))
			plt.legend()
			plt.subplot(gs[1, 0])
			absMax = np.max(np.abs(g[i*100:(i+1)*100, :]))
			plt.imshow(g[i*100:(i+1)*100, :].T, interpolation='nearest', cmap = cm, vmin = -absMax, vmax = absMax, aspect = 'auto')
			plt.gca().set_yticks([0,1,2,3,4,5,6])
			plt.gca().set_yticklabels(genes[:7])
			cax = plt.subplot(gs[1, 1])
			plt.colorbar(cax = cax)
			plt.subplot(gs[2, 0])
			plt.imshow(hist[i*100:(i+1)*100, j, ::-1].T, aspect = 'auto')
			plt.savefig("plots/selfdynamicNetwork_"+genes[j+3]+"_"+str(i)+"_t"+str(tstep)+".pdf")
			plt.clf()
			plt.close()
			
def plotSimpleResults(real, inferred, genes, tstep):
	ap = np.linspace(0, 1, 100)
	nSamples = real.shape[0]
	nGenes = real.shape[1]
	for i in xrange(nSamples/100):
		for j in xrange(nGenes):
			plt.title(genes[j+3]+" at time class "+str(tstep) + " for type " + str(i))
			plt.plot(ap, real[i*100:(i+1)*100,j], label='real')
			plt.plot(ap, inferred[i*100:(i+1)*100,j], label='mea_inf')
			plt.savefig("plots/fakedynamicNetwork_"+genes[j+3]+"_"+str(i)+"_t"+str(tstep)+".pdf")
			plt.clf()
			
def getProbFromModel(models, inputs, scale, samples, nBins):
	print "Calculating histograms...",
	nGenes = len(models)
	hist = np.zeros((inputs.shape[0], nGenes, nBins))
	for _ in xrange(samples):
		ri = inputs + np.random.normal(scale=scale, size=np.prod(inputs.shape)).reshape(inputs.shape)
		for gi in xrange(nGenes):
			models[gi].input_set.set_value(ri.astype('float32'))
			out = models[gi].evalNet()
			for i0 in xrange(hist.shape[0]):
				index = out[i0,0] * nBins
				if index >= nBins:
					index = nBins - 1
				hist[i0, gi, index] += 1
	print "done"
	return hist

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train a network')
	parser.add_argument('--f', dest='folder',
					type=str, default='/Users/tiago/Dropbox/workspace/evolveFlyNet/',
					help="folder")
	parser.add_argument('--nh', dest='nh',
					type=int, default=10,
					help="nr hidden neurons")
	
	args = parser.parse_args()
	data = dp.DataProcessor(args.folder)
	geneNames = data.getGeneNames()
	genes = geneNames[3:]
	
	control, target = loadData(data)
	'''four gap genes at time 0'''
	state = target[:, 0, :4]
	timeUnits = control.shape[1]
	input_set = theano.shared(np.ones((control.shape[0], control.shape[2]+state.shape[1])).astype('float32'))
	target_set = theano.shared(np.ones((state.shape[0], 1)).astype('float32'))
	models = []
	for g in genes:
		model = sn.SimpleNet()
		model.loadState("data/nets/"+g+"h"+str(args.nh)+"Net.nn",  input_set, target_set)
		models.append(model)
	
	for i in xrange(1, timeUnits):
		'''
		calculate output with previous net output as input
		'''
		inputSignal = np.concatenate((control[:,i-1,:], state[:, :4]), axis=1)
		input_set.set_value(inputSignal)
		newState = []
		for gi, gene in enumerate(genes):
			tSet = target[:,i-1,gi].reshape((200,1))
			target_set.set_value(tSet)
			newState.append(models[gi].evalNet())
			print i, gene, np.sqrt(models[gi].evalCost())
		state = np.array(newState).astype('float32')
		state = np.rollaxis(state, 1, 0)
		state = state.reshape((state.shape[0], state.shape[1]))
		'''
		compare with actual measurements as inputs
		'''
		realInputSignal = np.concatenate((control[:,i-1,:], target[:,i-1,:4]), axis=1)
		input_set.set_value(realInputSignal)
		compState = []
		for gi, gene in enumerate(genes):
			compState.append(models[gi].evalNet())
		compState = np.array(compState).astype('float32')
		compState = np.rollaxis(compState, 1, 0)
		'''
		calculate partial derivatives
		'''
		nSamples = realInputSignal.shape[0]
		nGenes = realInputSignal.shape[1]
		nOutputGenes = len(genes)
		derivatives = np.zeros((nOutputGenes, nSamples, nGenes))
		for gi, gene in enumerate(genes):
			for j in xrange(nSamples):
				derivatives[gi, j, :] = models[gi].evalGrad(j)
		
		hist = getProbFromModel(models, inputSignal, 0.1, 5000, 50)
		
		'''
		plot all this stuff
		'''
		plotResults(target[:,i-1,:], state, compState, derivatives, hist, geneNames, i+1)
		
	for i in xrange(1, timeUnits):
		'''
		compare with actual measurements as inputs
		'''
		fakeInputSignal = np.concatenate((control[:,i-1,:], target[:,i-1,:4]), axis=1)
		fakeInputSignal[:,-1] = 0
		input_set.set_value(fakeInputSignal)
		compState = []
		for gi, gene in enumerate(genes):
			compState.append(models[gi].evalNet())
		compState = np.array(compState).astype('float32')
		compState = np.rollaxis(compState, 1, 0)
		plotSimpleResults(target[:,i-1,:], compState, geneNames, i+1)
