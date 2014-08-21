import SimpleNet as sn
import os.path
import theano
import theano.sandbox.rng_mrg
import hashlib
import random
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt


class Lattice(object):
	def __init__(self, L=10):
		self.l_size = L
		self.main = np.zeros((L, L))

	def sample_with_location(self, values=None, flat=True):
		if values is None:
			values = self.main
		l_size = values.shape[0]
		m = l_size-1
		r_vector = np.zeros((values.shape[0], values.shape[1], 7))
		r_vector[:, :, 0] = values
		r_vector[:, :, 1] = np.roll(values, 1, axis=0)
		r_vector[0, :, 1] = r_vector[1, :, 1]
		r_vector[:, :, 2] = np.roll(values, -1, axis=0)
		r_vector[m, :, 2] = r_vector[m-1, :, 1]
		r_vector[:, :, 3] = np.roll(values, 1, axis=1)
		r_vector[:, 0, 3] = r_vector[:, 1, 1]
		r_vector[:, :, 4] = np.roll(values, -1, axis=1)
		r_vector[:, m, 4] = r_vector[:, m-1, 1]
		r_vector[:, :, 5] = np.tile(np.linspace(0, 1, l_size), (l_size, 1))
		r_vector[:, :, 6] = np.tile(np.linspace(0, 1, l_size), (l_size, 1)).T
		if flat:
			return np.reshape(r_vector, (l_size*l_size, 7))
		else:
			return r_vector

	def generate_training_set(self, factor=100, sigma_hi=0.1, sigma_lo=0.01):
		samples = self.sample_with_location()
		sample_set = np.tile(samples, (factor, 1))
		sample_set += np.hstack((
			np.random.normal(scale=sigma_hi, size=(sample_set.shape[0], 5)),
			np.random.normal(scale=sigma_lo, size=(sample_set.shape[0], 2))
		))
		sample_set[sample_set > 1] = 2 - sample_set[sample_set > 1]
		sample_set[sample_set < 0] *= -1
		target_set = np.tile(self.main.flatten(), factor)
		return sample_set, target_set

	def generate_unbiased_training_set(self, num_samples=100, sigma_hi=0.1, sigma_lo=0.01):
		"""
		Generate a training set making sure there are equal numbers of samples
		with zero and one values

		Keyword arguments:
		num_samples -- number of samples
		sigma_hi -- noise applied to lattice values
		sigma_lo -- noise applied to external inputs
		"""
		samples = self.sample_with_location()
		flat_main = self.main.flatten()
		ones_keys = np.where(flat_main == 1)[0]
		zeros_keys = np.where(flat_main == 0)[0]
		sample_set = np.zeros((num_samples, 7))
		for i in xrange(num_samples/2):
			o_ind = random.choice(ones_keys)
			z_ind = random.choice(zeros_keys)
			sample_set[2*i] = samples[o_ind]
			sample_set[2*i+1] = samples[z_ind]
		np.random.shuffle(sample_set)
		target_set = np.copy(sample_set[:, 0])
		sample_set += np.hstack((
			np.random.normal(scale=sigma_hi, size=(sample_set.shape[0], 5)),
			np.random.normal(scale=sigma_lo, size=(sample_set.shape[0], 2))
		))
		sample_set[sample_set > 1] = 2 - sample_set[sample_set > 1]
		sample_set[sample_set < 0] *= -1
		return sample_set, target_set

	def read_from_file(self, filename):
		with open(filename, 'r') as f:
			i = 0
			for line in f:
				vals = line.split(',')
				self.main[i, :] = vals
				i += 1
	
	def evaluate_lattice(self, model, values, prefix='net'):
		l_size = values.shape[0]
		samples = self.sample_with_location(values)
		prob = model.evalValueSet(samples)
		prob_lattice = np.reshape(prob, (l_size, l_size))
		plt.pcolor(
			np.linspace(0, 1, l_size+1),
			np.linspace(0, 1, l_size+1),
			prob_lattice,
			alpha=0.8,
			cmap=plt.get_cmap('Blues')
		)
		plt.savefig(("plots/{0}/lattice.pdf".format(prefix)))

	def evolve_network(self, model, init, dt, tot_time, prefix='net', ps=0.01):
		l_size = init.shape[0]
		pc = np.copy(init)
		for _ in xrange(int(tot_time/dt)):
			samples = self.sample_with_location(pc)
			samples += np.hstack((
				np.zeros((samples.shape[0], 5)),
				np.random.normal(scale=ps, size=(samples.shape[0], 2))
			))
			res = model.evalValueSet(samples)
			f = np.reshape(res.T, (l_size, l_size)) - pc
			pc += dt*f
		print np.sum(np.abs(pc-self.main))
		samples = self.sample_with_location(pc)
		res = model.evalValueSet(samples)
		print np.sum(np.power(np.reshape(res.T, (l_size, l_size))-pc,2))
		plt.pcolor(
			np.linspace(0, 1, l_size+1),
			np.linspace(0, 1, l_size+1),
			pc,
			alpha=0.8,
			cmap=plt.get_cmap('Blues')
		)
		plt.savefig("plots/{0}/evolution.pdf".format(prefix))

if __name__ == '__main__':
	lat_size = 10
	sigma_n = 0.6
	sigma_p = 0.01
	hidden_layer_size = 8
	l2_penalty = 1E-5
	num_generations = 300
	retrain = True
	pattern_name = 'diag'
	m = hashlib.md5()
	lat = Lattice(lat_size)
	lat.read_from_file("data/{0}.txt".format(pattern_name))
	config_string = "{0:03d}_{1:.3f}_{2:.3f}_{3:03d}_{4:.3e}_{5:05d}_{6}".format(
		lat_size, sigma_n, sigma_p, hidden_layer_size,
		l2_penalty, num_generations, pattern_name
	)
	m.update(config_string)
	config_hash = m.hexdigest()

	inputs, outputs = lat.generate_training_set(factor=1000, sigma_hi=sigma_n, sigma_lo=sigma_p)
	#inputs, outputs = lat.generate_unbiased_training_set(num_samples=100000, sigma_hi=sigma_n, sigma_lo=sigma_p)
	input_set = theano.shared(inputs.astype("float32"))
	output_set = theano.shared(outputs.reshape((outputs.shape[0], 1)).astype("float32"))
	model = sn.SimpleNet(seed=999888)
	net_name = "data/nets/{0}.nn".format(config_hash)
	if os.path.isfile(net_name) and not retrain:
		print 'found an existing function. loading {0}'.format(net_name)
		model.loadState(net_name, input_set, output_set, L2_reg=l2_penalty)
	else:
		model.createNewNet(
			inputs.shape[1],
			1,
			input_set,
			output_set,
			nHidden=hidden_layer_size,
			L2_reg=l2_penalty
		)
		model.trainDiagnostic(num_generations, inputs.shape[0], prefix="latticenet")
		model.saveState(net_name)

	model.visualizeWeights(prefix="latticenet")
	lat.evaluate_lattice(model, np.zeros((lat_size, lat_size)), prefix="latticenet")
	print model.evalCost(), model.evalRegCost()
	lat.evolve_network(model, np.zeros((lat_size, lat_size)), 0.05, 50, prefix="latticenet", ps=0.01)