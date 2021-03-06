'''
Created on Aug 15, 2013

@author: tiago
'''

import numpy as np
import re
import scipy.interpolate as interpolate
from copy import deepcopy

class DataProcessor(object):
	'''
	Reads the experimental files in the data folder and places them in
	a numpy array coreData. The array core data has five dimensions,
	organized in the following way:
	
	0 -- Temporal class [1-8]
	1 -- Measured gene ['bcd', 'cad', 'tll', 'gt', 'hb', 'kni', 'kr', 'eve']
	2 -- Experiment type [wt, dmtll]
	3 -- AP axis position [0 - 99]
	4 -- Measurement type [position, av. concentration, std. dev. of concentration]
	'''
	def __init__(self, folder = '', force = False):
		'''
		If we have a copy of the data array in numpy format, load that file
		Else create the data array from files
		'''
		self.genes = ['bcd', 'cad', 'tll', 'gt', 'hb', 'kni', 'kr', 'eve']
		
		if force:
			self.createDataFile(folder)
		else:
			try:
				self.coreData = np.load(folder+"data/npData.npy", None)
			except IOError:
				self.createDataFile(folder)
		self.normalizeCoreData()
			
	def createDataFile(self, folder):
		'''
		Loop over the relevant dimensions, and read the data from
		the corresponding file. Merge the data into the coreData array
		'''
		print "Reading data from experiment files..."
		self.pattern = re.compile('([-\d\.]+?) +?([-\d\.]+?) +?([-\d\.]+?) +?([-\d\.]+?)\n')
		data = []
		for i in range(1, 9):
			tData = []
			for g in self.genes:
				wtData = self.consumeFile(folder+'data/wt/wtg_'+g+'_t'+str(i)+'.100')
				dtData = self.consumeFile(folder+'data/dmtll/dmtllg_'+g+'_t'+str(i)+'.100')
				tData.append([wtData, dtData])
			data.append(tData)
		self.coreData = np.array(data, dtype = np.float32)
		np.save(folder+"data/npData.npy", self.coreData)
			
	def consumeFile(self, fname):
		'''
		Extract the data from an individual file
		'''
		try:
			with open(fname) as f:
				data = []
				for line in f:
					if line[0] == '#':
						continue
					res = self.pattern.search(line)
					if res is not None:
						n = [float(res.group(2)),float(res.group(3)),float(res.group(4))]
						data.append(n)
				return data
		except IOError:
			return np.zeros((100,3))
		
	def normalizeCoreData(self):
		'''
		Create an array with only the normalized average values of the measurements.
		The normalization is done per gene per experiment. The array norm data has 
		four dimensions, organized in the following way:
	
		0 -- Temporal class [1-8]
		1 -- Measured gene ['bcd', 'cad', 'tll', 'gt', 'hb', 'kni', 'kr', 'eve']
		2 -- Experiment type [wt, dmtll]
		3 -- AP axis position [0 - 99]
		'''
		self.normData = np.copy(self.coreData[:,:,:,:,1])
		self.normData = self.normData.clip(min = 0)
		for i in xrange(self.normData.shape[1]):
			for j in xrange(self.normData.shape[2]):
				mc = np.max(self.normData[:,i,j,:])
				if(mc > 0):
					self.normData[:,i,j,:] /= mc
	
	def sequencesPerCell(self):
		'''
		Return a rearranged data array, with less dimensions. Only average
		concentrations are selected, and the experiment type dimension
		is flattened. The remaining dimensions are:
		
		0 -- AP axis position [0 - 99 (twice)]
		1 -- Temporal class [1-8]
		2 -- Measured gene ['bcd', 'cad', 'tll', 'gt', 'hb', 'kni', 'kr', 'eve']
		'''
		tr = np.transpose(self.coreData, axes=(2, 3, 0, 1, 4))
		res = tr[:,:,:,:,1].flatten().reshape((200,8,8))
		return res
	
	def getGeneNames(self):
		'''
		Return a list with gene names ordered as in the data arrays
		'''
		return deepcopy(self.genes)
	
	def normalizedSequencesPerCell(self):
		'''
		Return a rearranged data array, with less dimensions. Only normalized
		average concentrations are selected, and the experiment type dimension
		is flattened. The remaining dimensions are:
		
		0 -- AP axis position [0 - 99 (twice)]
		1 -- Temporal class [1-8]
		2 -- Measured gene ['bcd', 'cad', 'tll', 'gt', 'hb', 'kni', 'kr', 'eve']
		'''
		tr = np.transpose(self.normData, axes=(2, 3, 0, 1))
		res = tr[:,:,:,:].flatten().reshape((200,8,8))
		return res
	
	def getInterpolatedInputOutputSequence(self, dt, exp_type=0):
		tr = np.transpose(self.normData, axes=(0, 3, 1, 2))
		nsp = tr[:,:,:, exp_type]
		'''Select last 5 genes as target values'''
		output_sequence = np.array(nsp[1:, :, 3:7], dtype='float32')
		'''Select first time point of last 5 genes as initial condition'''
		init_state = np.array(nsp[0, :, 3:7], dtype='float32')
		
		steps_per_unit = 1/dt
		num_units = output_sequence.shape[0]
		total_steps = num_units*steps_per_unit
		
		'''Create input genes array'''
		interp_genes = []
		for g in xrange(3):
			interp_genes.append(self.interpolateSingleGene(nsp[:,:,g], total_steps))
		
		interp_genes = np.array(interp_genes, dtype='float32')
		input_sequence = np.transpose(interp_genes, axes=(2, 1, 0))
		
		return input_sequence, output_sequence, init_state
	
	def interpolateSingleGene(self, data, totalSteps, plot=False):
		nt = data.shape[0] #n_time
		ns = data.shape[1] #n_space
		x = np.linspace(0, totalSteps, nt)
		y = np.linspace(0, 100, ns)
		spline = interpolate.RectBivariateSpline(x, y, data)
		x_big = np.linspace(0, totalSteps, totalSteps)
		result = np.array(spline(x_big, y)).transpose()
		return result.clip(min = 0)