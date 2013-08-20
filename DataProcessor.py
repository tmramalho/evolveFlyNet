'''
Created on Aug 15, 2013

@author: tiago
'''

import numpy as np
import re

class DataProcessor(object):
	'''
	classdocs
	'''
	def __init__(self, folder = '/Users/tiago/Dropbox/workspace/evolveFlyNet/', force = False):
		'''
		Constructor
		'''
		if force:
			self.createDataFile(folder)
		else:
			try:
				self.coreData = np.load(folder+"data/npData.npy", None)
			except IOError:
				self.createDataFile(folder)
			
	def createDataFile(self, folder):
		print "Reading data from experiment files..."
		self.pattern = re.compile('([-\d\.]+?) +?([-\d\.]+?) +?([-\d\.]+?) +?([-\d\.]+?)\n')
		genes = ['bcd', 'cad', 'tll', 'gt', 'hb', 'kni', 'kr', 'eve']
		data = []
		for i in range(1, 9):
			tData = []
			for g in genes:
				wtData = self.consumeFile(folder+'data/wt/wtg_'+g+'_t'+str(i)+'.100')
				dtData = self.consumeFile(folder+'data/dmtll/dmtllg_'+g+'_t'+str(i)+'.100')
				tData.append([wtData, dtData])
			data.append(tData)
		self.coreData = np.array(data, dtype = np.float32)
		np.save(folder+"data/npData.npy", self.coreData)
			
	def consumeFile(self, fname):
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