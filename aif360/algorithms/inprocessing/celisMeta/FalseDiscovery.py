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


class FalseDiscovery(General):

	def getExpectedGrad(self, dist_params, params, samples, mu,  z_0, z_1, a, b):
		u_1, u_2, l_1, l_2 = params[0], params[1], params[2], params[3]
		a, b = a[0], b[0]
		res1 = []
		res2 = []
		res3 = []
		res4 = []
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

				probc_m1_0 = prob_m1_0 / (prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1)
				probc_m1_1 = prob_m1_1 / (prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1)

				c_0 = prob_y_1 - 0.5
				c_1 = u_1 * (probc_m1_0 - a*prob_z_0) + u_2 * (probc_m1_1 - a*prob_z_1)
				c_2 = l_1 * (- probc_m1_0 + b*prob_z_0) + l_2 * (- probc_m1_1 + b*prob_z_1)

				t = math.sqrt((c_0 + c_1 + c_2)*(c_0 + c_1 + c_2) + mu*mu)
				t1 = (c_0 + c_1 + c_2) * (probc_m1_0 - a*prob_z_0)/t
				t2 = (c_0 + c_1 + c_2) * (probc_m1_1 - a*prob_z_1)/t
				t3 = (c_0 + c_1 + c_2) * (- probc_m1_0 + b*prob_z_0)/t
				t4 = (c_0 + c_1 + c_2) * (- probc_m1_1 + b*prob_z_1)/t
				#print(t1,t2)
				res1.append(t1)
				res2.append(t2)
				res3.append(t3)
				res4.append(t4)

		return [np.mean(res1), np.mean(res2), np.mean(res3), np.mean(res4)]

	def getValueForX(self, dist_params, a,b, params, samples,  z_0, z_1, x, flag):
				u_1, u_2, l_1, l_2 = params[0], params[1], params[2], params[3]
				#print (params)
				a, b = a[0], b[0]

				temp = np.append(np.append(x, 1), 1)
				prob_1_1 = ut.getProbability(dist_params, temp)

				temp = np.append(np.append(x, -1), 1)
				prob_m1_1 = ut.getProbability(dist_params, temp)

				temp = np.append(np.append(x, 1), 0)
				prob_1_0 = ut.getProbability(dist_params, temp)

				temp = np.append(np.append(x, -1), 0)
				prob_m1_0 = ut.getProbability(dist_params, temp)

				if (prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1) == 0:
					print("Probability is 0.\n")
					return 0

				prob_y_1 = (prob_1_1 + prob_1_0) / (prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1)
				#print(prob_y_1)

				prob_z_0 = (prob_m1_0 + prob_1_0) / (prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1)
				prob_z_1 = (prob_m1_1 + prob_1_1) / (prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1)


				probc_m1_0 = prob_m1_0 / (prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1)
				probc_m1_1 = prob_m1_1 / (prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1)

				c_0 = prob_y_1 - 0.5
				c_1 = u_1 * (probc_m1_0 - a*prob_z_0) + u_2 * (probc_m1_1 - a*prob_z_1)
				c_2 = l_1 * (- probc_m1_0 + b*prob_z_0) + l_2 * (- probc_m1_1 + b*prob_z_1)
				if flag==1:
					print(c_0, c_1, c_2, prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1)

				# c_1 = prob_z_0/z_0
				# c_2 = prob_z_1/z_1

				t = c_0 + c_1 + c_2
				return t

	def getFuncValue(self, dist_params, a,b, params, samples,  z_0, z_1):
		res = []
		for x in samples:
				t = abs(self.getValueForX(dist_params, a,b, params, samples,  z_0, z_1, x, 0))
				res.append(t)

		exp = np.mean(res)
		return exp

	def getNumOfParams(self):
		return 4

	def getGamma(self, y_test, y_res, x_control_test):
			pos_0 = 0
			pos_1 = 0

			z1_0 = 0
			z1_1 = 0
			for j in range(0,len(y_test)):
				result = y_res[j]

				if result == 1 and x_control_test[j] == 0:
					z1_0 += 1
				if result == 1 and x_control_test[j] == 1:
					z1_1 += 1

				actual = y_test[j]
				if result == 1 and actual == -1 and x_control_test[j] == 0:
					pos_0 += 1
				if result == 1 and actual == -1 and x_control_test[j] == 1:
					pos_1 += 1

			pos_0 = float(pos_0)/z1_0
			pos_1 = float(pos_1)/z1_1
			if pos_0 == 0 or pos_1 == 0:
				return 0
			else:
				return min(pos_0/pos_1 , pos_1/pos_0)


if __name__ == '__main__':
	obj = FalseDiscovery()
	obj.testPreprocessedData()
	#obj.testSyntheticData()
