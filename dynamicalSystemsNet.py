'''
Created on Dec 26, 2013

@author: tiago
'''

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import SimpleNet as sn
import scipy.interpolate as ip

def norm(x):
	max_x = np.max(x)
	min_x = np.min(x)
	return 2/(max_x-min_x)*x+1-max_x*2/(max_x-min_x), max_x, min_x

def unnorm(y, maxy, miny):
	return ((maxy-miny)*(y-1) + 2*maxy) / 2
	
if __name__ == '__main__':
	dt = 0.1
	r = 1
	series = []
	truef = []
	N = 100
	for x in [-0.3, 0, 0.5]:
		for y in [-0.1, 0.4, 0.2]:
			inputs = [[x,y]]
			a = r*(1-x*x)*y-x
			truef.append([y,a])
			for i in xrange(N-1):
				a = r*(1-x*x)*y-x
				x += y*dt+0.5*a*dt*dt
				at = r*(1-x*x)*y-x
				y += (a + at)*0.5*dt
				inputs.append([x,y])
				truef.append([y,a])
			series.append(inputs)
	truef = np.array(truef)
	series = np.array(series)
	series += np.random.normal(scale=0.1, size=series.shape)
	t = np.linspace(0,1,N)
	ox = []
	oy = []
	for i in xrange(9):
		ix = ip.UnivariateSpline(t, series[i,:,0], s=2)
		ox.append(ix(t, 1)*dt)
		iy = ip.UnivariateSpline(t, series[i,:,1], s=2)
		oy.append(iy(t, 1)*dt)
	ox = np.array(ox).flatten()
	oy = np.array(oy).flatten()
	ox_norm, max_ox, min_ox = norm(ox)
	oy_norm, max_oy, min_oy = norm(oy)
	inputs = np.array([series[:,:,0].flatten(), series[:,:,1].flatten()]).transpose()
	outputs = np.array([ox_norm, oy_norm]).transpose()
	input_set = theano.shared(inputs.astype("float32"))
	target_set = theano.shared(outputs.astype("float32"))
	model = sn.SimpleNet()
	model.createNewNet(inputs.shape[1], outputs.shape[1], input_set, target_set, L1_reg = 0, activation=T.tanh)
	model.train(3000, inputs.shape[0])
	res = model.evalNet()
	plt.subplot(211)
	plt.plot(ox)
	plt.plot(unnorm(res[:,0], max_ox, min_ox))
	plt.plot(truef[:,0])
	plt.subplot(212)
	plt.plot(oy)
	plt.plot(unnorm(res[:,1], max_oy, min_oy))
	plt.plot(truef[:,1])
	plt.show()
	plt.clf()
	cost = np.power(ox - unnorm(res[:,0], max_ox, min_ox), 2) + np.power(oy - unnorm(res[:,1], max_oy, min_oy), 2)
	plt.scatter(series[:,:,0], series[:,:,1], c=cost, cmap=plt.cm.RdBu)
	plt.colorbar()
	plt.show()
	plt.clf()
	
	x = xt = 0.5
	y = yt = 0.2
	dt = 0.1
	results = [[x,y]]
	true = [[xt,yt]]
	comp = []
	for i in xrange(N-1):
		val = np.array([x,y]).astype("float32")
		b = model.evalValue(val)
		x += dt*unnorm(b[0], max_ox, min_ox)
		y += dt*unnorm(b[1], max_oy, min_oy)
		results.append([x,y])
		a = r*(1-xt*xt)*yt-xt
		xt += yt*dt+0.5*a*dt*dt
		at = r*(1-xt*xt)*yt-xt
		yt += (a + at)*0.5*dt
		true.append([xt,yt])
		comp.append([[yt,a],b])
	comp = np.array(comp)
	results = np.array(results)
	true = np.array(true)
	plt.subplot(311)
	plt.plot(true[:, 0])
	plt.plot(results[:, 0])
	plt.plot(series[8,:,0])
	plt.subplot(312)
	plt.plot(comp[:, :, 0])
	plt.subplot(313)
	plt.plot(comp[:, :, 1])
	plt.show()