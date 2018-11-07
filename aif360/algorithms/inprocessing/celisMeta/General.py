from __future__ import division

import os,sys
from scipy.stats import multivariate_normal
import scipy.stats as st
import numpy as np
import math
from sklearn.mixture import GaussianMixture
import logging
from . import utils as ut

# This is the class with the general functions of the algorithm.
# For different fairness metrics, the objective function of the optimization problem is different and hence needs different implementations.
# The fairness-metric specific methods need to extend this class and implement the necessary functions
class General:

	# Used in gradient descent algorithm. Returns the value of gradient at any step.
	def getExpectedGrad(self, dist_params, params, samples, mu,  z_0, z_1, a, b):
		raise NotImplementedError("Expected gradient function not implemented")
		return []

	# Returns the threshold value at any point.
	def getValueForX(self, dist_params, a,b, params, samples,  z_0, z_1, x, flag):
		raise NotImplementedError("GetValueForX function not implemented")
		return 0

	# Returns the value of the objective function for given parameters.
	def getFuncValue(self, dist_params, a,b, params, samples,  z_0, z_1):
		raise NotImplementedError("Value function not implemented")
		return 0

	def getNumOfParams(self):
		raise NotImplementedError("Specify number of params")
		return 0

	def getRange(self, eps, tau):
		span = []
		L = math.ceil(tau/eps)
		for i in range(1, int(L+1), 10):
			a = (i-1) * eps
			b = (i) * eps / tau
			if b > 1:
				b = 1.0

			span.append(([a, -1],[b, -1]))
		return span

	def getGamma(self, y_test, y_res, x_control_test):
		raise NotImplementedError("Gamma function not implemented")
		return 0

	def getStartParams(self, i):
		num = self.getNumOfParams()
		return [i] * num

	# Gradient Descent implementation for the optimizing the objective function.
	# Note that one can alternately also use packages like CVXPY here.
	# Here we use decaying step size. For certain objectives, constant step size might be better.
	def gradientDescent(self, dist_params, a, b, samples, z_0, z_1):
		mu = 0.01
		minVal = 100000000
		size = self.getNumOfParams()

		minParam = [0] * size

		for i in range(1,10):
			params = self.getStartParams(i)
			for k in range(1,50):
				grad = self.getExpectedGrad(dist_params, params, samples, mu, z_0, z_1, a, b)
				for j in range(0, len(params)):
					params[j] = params[j] - 1/k * grad[j]
				funcVal = self.getFuncValue(dist_params, a,b, params, samples, z_0, z_1)
				if funcVal < minVal:
					minVal, minParam = funcVal, params

		return minParam

	# Returns the model given the training data and input tau.
	def getModel(self, tau, x_train, y_train, x_control_train):
		if tau == 0:
			return self.getUnbiasedModel(x_train, y_train, x_control_train)

		dist_params, dist_params_train =  ut.getDistribution(x_train, y_train, x_control_train)
		eps = 0.01
		L = math.ceil(tau/eps)
		z_1 = sum(x_control_train)/(float(len(x_control_train)))
		z_0 = 1 - z_1
		p, q  = [0,0],[0,0]
		paramsOpt, samples = [], []
		maxAcc = 0
		maxGamma = 0

		span = self.getRange(eps, tau)
		for (a,b) in span:
			acc, gamma = 0, 0
			#print("-----",a,b)
			samples = ut.getRandomSamples(dist_params_train)

			#try :
			params = self.gradientDescent(dist_params, a, b, samples, z_0, z_1)
			#print(params)
			y_res = []

			for x in x_train:
				t = self.getValueForX(dist_params, a,b, params, samples,  z_0, z_1, x, 0)
				if t > 0 :
					y_res.append(1)
				else:
					y_res.append(-1)

			acc = ut.getAccuracy(y_train, y_res)
			gamma = self.getGamma(y_train, y_res, x_control_train)
			#print(acc, gamma)

			if maxAcc < acc and gamma >= tau - 0.2:
				maxGamma = gamma
				maxAcc = acc
				p = a
				q = b
				paramsOpt = params

		print("Training Accuracy: ", maxAcc, ", Training gamma: ", maxGamma)
		def model(x):
			return self.getValueForX(dist_params, p, q, paramsOpt, samples,  z_0, z_1, x, 0)

		return model

	def getUnbiasedModel(self, x_train, y_train, x_control_train):
		dist_params, dist_params_train =  ut.getDistribution(x_train, y_train, x_control_train)
		eps = 0.01
		z_1 = sum(x_control_train)/(float(len(x_control_train)))
		z_0 = 1 - z_1
		p, q  = [0,0],[0,0]
		params = [0]*self.getNumOfParams()
		samples = ut.getRandomSamples(dist_params_train)

		def model(x):
			return self.getValueForX(dist_params, p, q, params, samples,  z_0, z_1, x, 0)

		return model

	def processGivenData(self, tau, x_train, y_train, x_control_train, x_test, y_test, x_control_test, dist_params, dist_params_train):
		model = self.getModel(tau, x_train, y_train, x_control_train)

		y_test_res = []
		for x in x_test:
				#t = self.getValueForX(dist_params, p, q, paramsOpt, samples,  z_0, z_1, x, 0)
				t = model(x)
				if t > 0 :
					y_test_res.append(1)
				else:
					y_test_res.append(-1)
		#f.write(str(tau) + " " + str(self.getGamma(y_test, y_test_res, x_control_test)) + " " + str(ut.getAccuracy(y_test, y_test_res)) + "\n")
		return y_test_res

	def test_given_data(self, x_train, y_train, x_control_train, x_test, y_test, x_control_test, sensitive_attrs, tau):
		attr = sensitive_attrs[0]
		x_control_train = x_control_train[attr]
		x_control_test = x_control_test[attr]

		l = len(y_train)


		#print(mean, cov)

		return self.processGivenData(tau, x_train, y_train, x_control_train, x_test, y_test, x_control_test, [], [])

	global getData

	def testPreprocessedData(self):
		x_train, y_train, x_control_train, x_control_test, x_test, y_test = ut.getData()
		#checkNormalFit(x_train, y_train, x_control_train)

		for i in range(1,11):
			try :
				tau = i/10.0
				print("Tau : ", tau)
				y_res = self.processGivenData(tau, x_train, y_train, x_control_train, x_test, y_test, x_control_test, [], [])
				ut.getStats(y_test, y_res, x_control_test)
				print("\n")
			except Exception as e:
				logging.exception(str(tau) + " failed\n" + str(e))

	def testSyntheticData(self):
		#A,S,F = [],[],[]
		x_train, y_train, x_control_train, x_control_test, x_test, y_test = ut.getData()
		dist_params, dist_params_train =  ut.getDistribution(x_train, y_train, x_control_train)

		mean, cov, meanT, covT = dist_params["mean"], dist_params["cov"], dist_params_train["mean"], dist_params_train["cov"]
		#print(mean)
		meanN = [0] * len(mean)
		covN = np.identity(len(mean))

		#clf = GaussianMixture(n_components=2, covariance_type='full')
		means = [mean, meanN]
		covariances = [cov, covN]
		lw = float(sys.argv[2])
		weights = [1-lw, lw]

		#for i in range(0,4):
		LR, LE = len(y_train), len(y_test)
		train, test = [],[]
		for i in range(0, LR):
			j = np.random.choice([0,1], p=weights)
			seed = np.random.randint(10)
			train.append(multivariate_normal(means[j], covariances[j], allow_singular=1).rvs(size=1, random_state=seed))
		for i in range(0, LE):
			j = np.random.choice([0,1], p=weights)
			seed = np.random.randint(10)
			test.append(multivariate_normal(means[j], covariances[j], allow_singular=1).rvs(size=1, random_state=seed))

		x_train, y_train, x_control_train = [], [], []
		for t in train:
			x_train.append(t[:-2])
			if t[len(t)-2] < 0:
				y_train.append(-1)
			else:
				y_train.append(1)
			#y_train.append(t[len(t)-2])
			if t[len(t)-1] < 0.5:
				x_control_train.append(0)
			else:
				x_control_train.append(1)

		x_control_test, x_test, y_test = [], [], []
		for t in test:
			x_test.append(t[:-2])
			if t[len(t)-2] < 0:
				y_test.append(-1)
			else:
				y_test.append(1)
			if t[len(t)-1] < 0.5:
				x_control_test.append(0)
			else:
				x_control_test.append(1)

		#print(x_train, y_train, x_control_train)
		y_res = self.processGivenData(0.9, x_train, y_train, x_control_train, x_test, y_test, x_control_test, dist_params, dist_params_train)
		acc, sr, fdr = ut.getStats(y_test, y_res, x_control_test)
		print("Acc: ", acc, " SR: ", sr, " FDR: ", fdr)

		#print("\n", np.mean(A), np.std(A), np.mean(S), np.std(S), np.mean(F), np.std(F))
