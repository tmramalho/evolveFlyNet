'''
Created on Sep 6, 2013

@author: tiago
'''

import DataProcessor as dp
import SimpleNet as sn
import theano
import numpy as np
import matplotlib.pyplot as plt
import argparse

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

def loadData(data, gene = 'gt', timeClass = None):
	print "Creating numeric arrays..."
	nsp = data.normalizedSequencesPerCell()
	geneNames = data.getGeneNames()
	#plotData(nsp, geneNames)
	
	'''
	We dont need eve for the gap genes
	'''
	if gene != 'eve':
		geneNames.remove('eve')
	
	samplesByGene = dict()
	
	if timeClass == None:
		for i,g in enumerate(geneNames):
			samplesByGene[g] = nsp[:,:,i].flatten()
	else:
		for i,g in enumerate(geneNames):
			samplesByGene[g] = nsp[:,timeClass,i].flatten()

	target = samplesByGene[gene]
	del samplesByGene[gene]
	
	inputs = np.array(samplesByGene.values()).swapaxes(0, 1)
	
	return inputs, target

def plotResults(real, inferred, nSamples, index, model):
	ap = np.linspace(0, 1, nSamples)
	plt.plot(ap, real)
	plt.plot(ap, inferred)
	plt.savefig("plots/staticTest"+str(index)+".pdf")
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
		plt.savefig("plots/weightsLayer"+str(i/2)+"_"+str(i)+"_"+str(index)+".pdf")
		plt.clf()
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train a network')
	parser.add_argument('--ne', dest='n',
					type=int, default=1000,
					help="number of training epochs")
	parser.add_argument('--g', dest='gene',
					type=str, default='gt', choices=['gt', 'hb', 'kni', 'kr', 'eve'],
					help="target gene")
	parser.add_argument('--f', dest='folder',
					type=str, default='',
					help="folder")
	
	args = parser.parse_args()
	data = dp.DataProcessor(args.folder)
	
	inputs, target = loadData(data, args.gene, 0)
	input_set = theano.shared(inputs)
	target_set = theano.shared(target.reshape(target.shape[0], 1))	
	model = sn.SimpleNet()
	model.createNewNet(inputs.shape[1], 1, input_set, target_set, L1_reg = 0.0001)
	model.train(args.n, inputs.shape[0])
	plotResults(target, model.evalNet(), inputs.shape[0], 0, model)

	for n in xrange(7):
		inputs, target = loadData(data, args.gene, n + 1)
		input_set.set_value(inputs)
		target_set.set_value(target.reshape(target.shape[0], 1))
		model.train(args.n, inputs.shape[0])
		plotResults(target, model.evalNet(), inputs.shape[0], n + 1, model)
