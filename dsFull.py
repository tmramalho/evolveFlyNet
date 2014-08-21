'''
Created on Jan 2, 2014

@author: tiago
'''

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import SimpleNet as sn
import scipy.interpolate as ip
from DataProcessor import DataProcessor
from mpl_toolkits.mplot3d import Axes3D

class FullSystem(object):
	'''
	Creates the data representation and manages the training
	'''


	def __init__(self):
		'''
		Load data
		'''
		self.dp = DataProcessor()
		
	def normalize(self, vals, w, z):
		min_val = np.min(vals)
		max_val = np.max(vals)
		a = (z-w)/(max_val-min_val)
		b = (max_val*w-min_val*z)/(max_val-min_val)
		
		return a*vals + b, max_val, min_val
	
	def createDatasetForGene(self, gene_ind, plot = False):
		if gene_ind not in [3,4,5,6,7]:
			raise Exception("Wrong gene")
		'''use only wt data for now'''
		data = self.dp.normData[:,:,0,:]
		x_range = np.linspace(0, data.shape[2]-1, data.shape[2])
		t_range = np.linspace(0, data.shape[0]-1, data.shape[0])
		xv, tv = np.meshgrid(x_range, t_range)
		x = xv.flatten()
		t = tv.flatten()
		z = data[:,gene_ind,:].flatten()
		spdat = ip.bisplrep(x,t,z,s=5)
		t_der = ip.bisplev(x_range, t_range, spdat, dx=0, dy=1)
		x_der2 = ip.bisplev(x_range, t_range, spdat, dx=2, dy=0)
		input_list = []
		for g in xrange(7):
			input_list.append(data[:,g,:].flatten())
		input_list.append(x_der2.T.flatten())
		input_list = np.rollaxis(np.array(input_list), 1, 0)
		output_list, self.omax, self.omin = self.normalize(t_der.T.flatten(), -0.9, 0.9)
		
		if plot is True:
			fig = plt.figure()
			ax = fig.add_subplot(221, projection='3d')
			ax.plot_surface(xv, tv, t_der.T)
			ax = fig.add_subplot(222, projection='3d')
			ax.plot_surface(xv, tv, x_der2.T)
			ax = fig.add_subplot(223, projection='3d')
			x_range = np.linspace(0, data.shape[2]-1, 200)
			t_range = np.linspace(0, data.shape[0]-1, 200)
			xv, tv = np.meshgrid(x_range, t_range)
			plt_data = ip.bisplev(x_range, t_range, spdat)
			ax.plot_surface(xv, tv, plt_data.T)
			ax = fig.add_subplot(224)
			ax.hist(t_der.flatten(), bins=40)
			plt.show()
			exit()
		
		return input_list, output_list
		
if __name__ == "__main__":
	fs = FullSystem()
	in_list, out_list = fs.createDatasetForGene(3, False)
	out_list = out_list.reshape(out_list.shape[0],1)
	input_set = theano.shared(in_list.astype("float32"))
	target_set = theano.shared(out_list.astype("float32"))
	model = sn.SimpleNet(seed=None)
	model.createNewNet(in_list.shape[1], out_list.shape[1], input_set,
					target_set, L1_reg = 0.01, activation=T.tanh, 
					nHidden=[25,10], batch_size=40)
	diagn = model.trainDiagnostic(145000, in_list.shape[0])
	model.visualizeWeights()
	plt.plot(diagn)
	plt.savefig("plots/scoreEvolution.pdf")
	plt.clf()
	res = model.evalNet()
	plt.plot(res)
	plt.plot(out_list)
	plt.savefig("plots/valueComparison.pdf")
	model.saveState("results/dsNet.npy")