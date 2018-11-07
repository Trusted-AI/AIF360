from __future__ import division

import os,sys
from scipy.stats import multivariate_normal
import scipy.stats as st
import numpy as np
import math

import site
site.addsitedir('.')

from .General import *
from . import utils as ut


class StatisticalRate(General):

	def getExpectedGrad(self, dist_params, params, samples, mu,  z_0, z_1, a, b):
		a, b = a[0], b[0]
		l_1, l_2 = params[0], params[1]
		res1 = []
		res2 = []
		for x in samples:
			temp = np.append(np.append(x, 1), 1)
			prob_1_1 = ut.getProbability(dist_params, temp)

			temp = np.append(np.append(x, -1), 1)
			prob_m1_1 = ut.getProbability(dist_params, temp)

			temp = np.append(np.append(x, 1), 0)
			prob_1_0 = ut.getProbability(dist_params, temp)

			temp = np.append(np.append(x, -1), 0)
			prob_m1_0 = ut.getProbability(dist_params, temp)


			prob_y_1 = (prob_1_1 + prob_1_0) / (prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1)
			#print(prob_y_1)

			prob_z_0 = (prob_m1_0 + prob_1_0) / (prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1)
			prob_z_1 = (prob_m1_1 + prob_1_1) / (prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1)

			c_0 = prob_y_1 - 0.5
			c_1 = prob_z_0/z_0
			c_2 = prob_z_1/z_1

			t = math.sqrt((c_0 + c_1*l_1 + c_2*l_2)*(c_0 + c_1*l_1 + c_2*l_2) + mu*mu)
			t1 = (c_0 + c_1*l_1 + c_2*l_2) * c_1/t
			t2 = (c_0 + c_1*l_1 + c_2*l_2) * c_2/t
			#print(t1,t2)
			res1.append(t1)
			res2.append(t2)

		exp1 = np.mean(res1)
		exp2 = np.mean(res2)
		dl1 = exp1 - b + (b-a)/2 + (b-a)* l_1 / (2* math.sqrt(l_1*l_1 + mu*mu))
		dl2 = exp2 - b + (b-a)/2 + (b-a)* l_2 / (2* math.sqrt(l_2*l_2 + mu*mu))
		return [dl1, dl2]

	def getValueForX(self, dist_params, a,b, params, samples,  z_0, z_1, x, flag):
			a, b = a[0], b[0]
			l_1, l_2 = params[0], params[1]

			temp = np.append(np.append(x, 1), 1)
			prob_1_1 = ut.getProbability(dist_params, temp)

			temp = np.append(np.append(x, -1), 1)
			prob_m1_1 = ut.getProbability(dist_params, temp)

			temp = np.append(np.append(x, 1), 0)
			prob_1_0 = ut.getProbability(dist_params, temp)

			temp = np.append(np.append(x, -1), 0)
			prob_m1_0 = ut.getProbability(dist_params, temp)
			if (prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1) == 0:
				#print("Probability is 0.\n")
				return 0


			prob_y_1 = (prob_1_1 + prob_1_0) / (prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1)
			#print(prob_y_1)

			prob_z_0 = (prob_m1_0 + prob_1_0) / (prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1)
			prob_z_1 = (prob_m1_1 + prob_1_1) / (prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1)

			c_0 = prob_y_1 - 0.5
			c_1 = prob_z_0/z_0
			c_2 = prob_z_1/z_1
			if flag==1:
				print(c_0, c_1, c_2, prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1)

			t = c_0 + c_1*l_1 + c_2*l_2
			return t

	def getFuncValue(self, dist_params, a,b, params, samples,  z_0, z_1):
		res = []
		for x in samples:
				t = abs(self.getValueForX(dist_params, a,b, params, samples,  z_0, z_1, x, 0))
				res.append(t)

		l_1 = params[0]
		l_2 = params[1]
		a, b = a[0], b[0]

		exp = np.mean(res)
		result = exp - b*l_1 - b*l_2
		if l_1 > 0 :
			result += (b-a)*l_1
		if l_2 > 0 :
			result += (b-a)*l_2

		return result

	def getNumOfParams(self):
		return 2

	def getStartParams(self, i):
		num = self.getNumOfParams()
		return [i-5] * num

	def getGamma(self, y_test, y_res, x_control_test):
			pos_0 = 0
			pos_1 = 0

			z1_0 = 0
			z1_1 = 0
			for j in range(0,len(y_test)):
				result = y_res[j]
				actual = y_test[j]

				if x_control_test[j] == 0:
					z1_0 += 1
				if x_control_test[j] == 1:
					z1_1 += 1

				if result == 1 and x_control_test[j] == 0:
					pos_0 += 1
				if result == 1 and x_control_test[j] == 1:
					pos_1 += 1


			pos_0 = float(pos_0)/z1_0
			pos_1 = float(pos_1)/z1_1
			if pos_0 == 0 or pos_1 == 0:
				return 0
			else:
				return min(pos_0/pos_1 , pos_1/pos_0)


if __name__ == '__main__':
	obj = StatisticalRate()
	obj.testPreprocessedData()
	#obj.testSyntheticData()
