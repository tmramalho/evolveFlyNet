'''
Created on Sep 6, 2013

@author: tiago
'''

import DataProcessor as dp
import SimpleNet as sn
import theano
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
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

def loadData(data, gene = 'gt', smoothing = False, timeLabel = False):
	print "Creating numeric arrays..."
	nsp = data.normalizedSequencesPerCell()
	geneNames = data.getGeneNames()
	#plotData(nsp, geneNames)
	
	timeUnits = nsp.shape[1]
	inputs = []
	targets = []
	g = geneNames.index(gene)
	
	if smoothing:
		for i in xrange(1,timeUnits):
			'''all genes except eve at previous timestep are inputs'''
			targets.append(nsp[:100, i, g])
			inputs.append(nsp[:100, i-1, :7])
			'''apply smoothing to the dtll datapoints'''
			b = sig.gaussian(10, 1)
			targetTll = sig.filtfilt(b/b.sum(), [1.0], nsp[100:, i, g], axis=0)
			targets.append(targetTll)
			inputTll = sig.filtfilt(b/b.sum(), [1.0], nsp[100:, i-1, :7], axis=0)
			inputs.append(inputTll)
	else:
		for i in xrange(1,timeUnits):
			'''all genes except eve at previous timestep are inputs'''
			targets.append(nsp[:, i, g])
			inputs.append(nsp[:, i-1, :7])
		
	targets = np.concatenate(targets).astype("float32")
	inputs = np.concatenate(inputs).astype("float32")
	
	if timeLabel:
		nt = inputs.shape[1]
		ns = inputs.shape[0]
		l = np.linspace(0, 1, nt)
		l = np.tile(l, (ns/nt, 1)).T.flatten().reshape((ns, 1))
		inputs = np.concatenate((inputs, l), axis=1).astype("float32")

	return inputs, targets

def plotResults(real, inferred, inputs, name, model, data):
	ap = np.linspace(0, 1, 100)
	nSamples = inputs.shape[0]
	for i in xrange(nSamples/100):
		plt.subplot(211)
		plt.plot(ap, inputs[i*100:(i+1)*100])
		plt.subplot(212)
		plt.plot(ap, real[i*100:(i+1)*100], label='real')
		plt.plot(ap, inferred[i*100:(i+1)*100], label='inferred')
		tClass = i/2
		expType = i%2
		plt.title(name+" at time class "+str(tClass) + " for type " + str(expType))
		plt.legend()
		plt.savefig("plots/dynamicNetwork"+name+"_"+str(i)+".pdf")
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
		plt.savefig("plots/dynamicNetWeightsLayer"+str(i/2)+"_"+str(i%2)+"_"+name+".pdf")
		plt.clf()

def trainNetworkForGene(args, gene, nh, sigma=None):
	inputs, target = loadData(data, gene)
	input_set = theano.shared(inputs)
	target_set = theano.shared(target.reshape(target.shape[0], 1))	
	model = sn.SimpleNet()
	model.createNewNet(inputs.shape[1], 1, input_set, target_set, L1_reg = 0.0001, nHidden = [nh,3])
	if sigma is None:
		model.train(args.n, inputs.shape[0])
	else:
		model.jitterTrain(args.n, inputs.shape[0], sigma)
	plotResults(target, model.evalNet(), inputs, gene, model, data)
	model.saveState("data/nets/"+gene+"h"+str(nh)+"Net.nn")

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train a network')
	parser.add_argument('--ne', dest='n',
					type=int, default=1000,
					help="number of training epochs")
	parser.add_argument('--g', dest='gene',
					type=str, default='gt', choices=['gt', 'hb', 'kni', 'kr', 'eve'],
					help="target gene")
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
	
	if args.scan:
		for g in ['gt', 'hb', 'kni', 'kr']:
			#for h in [5,10,20]:
			print "Training network for", g, "with neurons", 10
			trainNetworkForGene(args, g, args.nh, sigma = args.sigma)
	else:
		trainNetworkForGene(args, args.gene, args.nh, sigma = args.sigma)