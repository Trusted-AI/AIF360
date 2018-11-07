from __future__ import division

import os,sys
from scipy.stats import multivariate_normal
import scipy.stats as st
import numpy as np
import math
from sklearn.mixture import GaussianMixture
import logging

def getDistribution(x_train, y_train, x_control_train):
	train = list(x_train)
	for i in range(0,len(train)):
		train[i] = np.append(train[i], y_train[i])
		train[i] = np.append(train[i], x_control_train[i])

	#print(train)

	mean = np.mean(train, axis=0)
	cov = np.cov(train, rowvar=0)
	clf = GaussianMixture(n_components=2, covariance_type='full')
	model = clf.fit(train)

	dist_params = {"mean":mean, "cov":cov, "model":model}

	mean_train = np.mean(x_train, axis=0)
	cov_train = np.cov(x_train, rowvar=0)
	clf_train = GaussianMixture(n_components=2, covariance_type='full')
	model_train = clf_train.fit(list(x_train))

	#print(train, train_label, train_label==train)

	dist_params_train = {"mean":mean_train, "cov":cov_train, "model":model_train}
	return dist_params, dist_params_train

def getProbability(dist_params, x):
	mean, cov = dist_params["mean"], dist_params["cov"]
	return multivariate_normal.pdf(x, mean=mean, cov=cov, allow_singular=1)

def getRandomSamples(dist_params_train):
	mean, cov, model = dist_params_train["mean"], dist_params_train["cov"], dist_params_train["model"]
	return multivariate_normal(mean, cov, allow_singular=1).rvs(size=20, random_state=12345)

def getAccuracy(y_test, y_res):
	total, fail = 0, 0
	for j in range(0,len(y_test)):
		result = y_res[j]
		actual = y_test[j]
		total += 1
		if actual != result:
			fail += 1

	return 1 - fail/(float(total))

def getStats(y_test, y_res, x_control_test):
	try:
		total_0 = 0
		total_1 = 0
		fail = 0
		pos_0 = 0
		pos_1 = 0
		for j in range(0,len(y_test)):
			result = y_res[j]
			actual = y_test[j]

			if x_control_test[j] == 0:
				total_0 += 1
				if actual == result:
					pos_0 += 1
			else:
				total_1 += 1
				if actual == result:
					pos_1 += 1

		total_0 = float(total_0)
		total_1 = float(total_1)

		#print("Accuracy DIFF: ", abs(pos_0/total_0 - pos_1/total_1))

		pos_0 = 0
		pos_1 = 0

		z1_0 = 0
		z1_1 = 0
		for j in range(0,len(y_test)):
			result = y_res[j]
			actual = y_test[j]

			if actual == 1 and x_control_test[j] == 0:
				z1_0 += 1
			if actual == 1 and x_control_test[j] == 1:
				z1_1 += 1

			if result == 1 and actual == 1 and x_control_test[j] == 0:
				pos_0 += 1
			if result == 1 and actual == 1 and x_control_test[j] == 1:
				pos_1 += 1


		pos_0 = float(pos_0)/z1_0
		pos_1 = float(pos_1)/z1_1
		# if pos_0 == 0 or pos_1 == 0:
		# 		print("Observed tau : 0")
		# else:
		# 	print("TPR DIFF : ", abs(pos_0 - pos_1))




		total = 0
		fail = 0
		pos_0 = 0
		pos_1 = 0
		for j in range(0,len(y_test)):
			result = y_res[j]
			actual = y_test[j]

			total += 1
			if actual != result:
				fail += 1


		print("Accuracy : ", fail, total, 1 - fail/(float(total)))
		acc = 1 - fail/(float(total))

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
				print("Observed tau : 0")
		else:
			print("SR tau : ", min(pos_0/pos_1 , pos_1/pos_0))
		sr = min(pos_0/pos_1 , pos_1/pos_0)


		pos_0 = 0
		pos_1 = 0

		z1_0 = 0
		z1_1 = 0
		for j in range(0,len(y_test)):
			result = y_res[j]
			actual = y_test[j]

			if actual == -1 and x_control_test[j] == 0:
				z1_0 += 1
			if actual == -1 and x_control_test[j] == 1:
				z1_1 += 1

			if result == 1 and actual == -1 and x_control_test[j] == 0:
				pos_0 += 1
			if result == 1 and actual == -1 and x_control_test[j] == 1:
				pos_1 += 1


		pos_0 = float(pos_0)/z1_0
		pos_1 = float(pos_1)/z1_1
		if pos_0 == 0 or pos_1 == 0:
				print("Observed tau : 0")
		else:
			print("FPR tau : ", min(pos_0/pos_1 , pos_1/pos_0))


		pos_0 = 0
		pos_1 = 0

		z1_0 = 0
		z1_1 = 0
		for j in range(0,len(y_test)):
			result = y_res[j]
			actual = y_test[j]

			if actual == 1 and x_control_test[j] == 0:
				z1_0 += 1
			if actual == 1 and x_control_test[j] == 1:
				z1_1 += 1

			if result == -1 and actual == 1 and x_control_test[j] == 0:
				pos_0 += 1
			if result == -1 and actual == 1 and x_control_test[j] == 1:
				pos_1 += 1


		pos_0 = float(pos_0)/z1_0
		pos_1 = float(pos_1)/z1_1
		if pos_0 == 0 or pos_1 == 0:
				print("Observed tau : 0")
		else:
			print("FNR tau : ", min(pos_0/pos_1 , pos_1/pos_0))

		pos_0 = 0
		pos_1 = 0

		z1_0 = 0
		z1_1 = 0
		for j in range(0,len(y_test)):
			result = y_res[j]
			actual = y_test[j]

			if actual == 1 and x_control_test[j] == 0:
				z1_0 += 1
			if actual == 1 and x_control_test[j] == 1:
				z1_1 += 1

			if result == 1 and actual == 1 and x_control_test[j] == 0:
				pos_0 += 1
			if result == 1 and actual == 1 and x_control_test[j] == 1:
				pos_1 += 1


		pos_0 = float(pos_0)/z1_0
		pos_1 = float(pos_1)/z1_1
		if pos_0 == 0 or pos_1 == 0:
				print("Observed tau : 0")
		else:
			print("TPR tau : ", min(pos_0/pos_1 , pos_1/pos_0))


		pos_0 = 0
		pos_1 = 0

		z1_0 = 0
		z1_1 = 0
		for j in range(0,len(y_test)):
			result = y_res[j]
			actual = y_test[j]

			if actual == -1 and x_control_test[j] == 0:
				z1_0 += 1
			if actual == -1 and x_control_test[j] == 1:
				z1_1 += 1

			if result == -1 and actual == -1 and x_control_test[j] == 0:
				pos_0 += 1
			if result == -1 and actual == -1 and x_control_test[j] == 1:
				pos_1 += 1


		pos_0 = float(pos_0)/z1_0
		pos_1 = float(pos_1)/z1_1
		if pos_0 == 0 or pos_1 == 0:
				print("Observed tau : 0")
		else:
			print("TNR tau : ", min(pos_0/pos_1 , pos_1/pos_0))


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

			if result == actual and x_control_test[j] == 0:
				pos_0 += 1
			if result == actual and x_control_test[j] == 1:
				pos_1 += 1


		pos_0 = float(pos_0)/z1_0
		pos_1 = float(pos_1)/z1_1
		if pos_0 == 0 or pos_1 == 0:
				print("Observed tau : 0")
		else:
			print("AR tau : ", min(pos_0/pos_1 , pos_1/pos_0))

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
				print("Observed tau : 0")
		else:
			print("FDR tau : ", min(pos_0/pos_1 , pos_1/pos_0))
		fdr = min(pos_0/pos_1 , pos_1/pos_0)

		pos_0 = 0
		pos_1 = 0

		z1_0 = 0
		z1_1 = 0
		for j in range(0,len(y_test)):
			result = y_res[j]

			if result == -1 and x_control_test[j] == 0:
				z1_0 += 1
			if result == -1 and x_control_test[j] == 1:
				z1_1 += 1

			actual = y_test[j]
			if result == -1 and actual == 1 and x_control_test[j] == 0:
				pos_0 += 1
			if result == -1 and actual == 1 and x_control_test[j] == 1:
				pos_1 += 1


		pos_0 = float(pos_0)/z1_0
		pos_1 = float(pos_1)/z1_1
		if pos_0 == 0 or pos_1 == 0:
				print("Observed tau : 0")
		else:
			print("FOR tau : ", min(pos_0/pos_1 , pos_1/pos_0))

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
			if result == 1 and actual == 1 and x_control_test[j] == 0:
				pos_0 += 1
			if result == 1 and actual == 1 and x_control_test[j] == 1:
				pos_1 += 1


		pos_0 = float(pos_0)/z1_0
		pos_1 = float(pos_1)/z1_1
		if pos_0 == 0 or pos_1 == 0:
				print("Observed tau : 0")
		else:
			print("PPR tau : ", min(pos_0/pos_1 , pos_1/pos_0))

		pos_0 = 0
		pos_1 = 0

		z1_0 = 0
		z1_1 = 0
		for j in range(0,len(y_test)):
			result = y_res[j]

			if result == -1 and x_control_test[j] == 0:
				z1_0 += 1
			if result == -1 and x_control_test[j] == 1:
				z1_1 += 1

			actual = y_test[j]
			if result == -1 and actual == -1 and x_control_test[j] == 0:
				pos_0 += 1
			if result == -1 and actual == -1 and x_control_test[j] == 1:
				pos_1 += 1


		pos_0 = float(pos_0)/z1_0
		pos_1 = float(pos_1)/z1_1
		if pos_0 == 0 or pos_1 == 0:
				print("Observed tau : 0")
		else:
			print("NPR tau : ", min(pos_0/pos_1 , pos_1/pos_0))

		return acc, sr, fdr
	except ZeroDivisionError:
		print("Stats inconclusive")

def getData():
	x_control_train = []
	x_train = []
	y_train = []
	x_control_test = []
	x_test = []
	y_test = []

	folder = sys.argv[1]
	temp = []
	with open(folder + "/x_train.txt") as f:
		temp = f.readlines()

	for line in temp:
		temp2 = line[:-1].split(' ')
		a = []
		for i in temp2[:-1]:
			a.append(float(i))
		x_train.append(np.array(a))
	temp = []
	with open(folder + "/x_test.txt") as f:
		temp = f.readlines()

	for line in temp:
		temp2 = line.split(' ')
		a = []
		for i in temp2[:-1]:
			a.append(float(i))
		x_test.append(np.array(a))

	temp = []
	with open(folder + "/y_train.txt") as f:
		temp = f.readlines()
	for line in temp:
		y_train.append(float(line))
	y_train = np.array(y_train)

	temp = []
	with open(folder + "/y_test.txt") as f:
		temp = f.readlines()
	for line in temp:
		y_test.append(float(line))
	y_test = np.array(y_test)

	temp = []
	with open(folder + "/x_control_train.txt") as f:
		temp = f.readlines()
	for line in temp:
		x_control_train.append(float(line))
	x_control_train = np.array(x_control_train)

	temp = []
	with open(folder + "/x_control_test.txt") as f:
		temp = f.readlines()
	for line in temp:
		x_control_test.append(float(line))
	x_control_test = np.array(x_control_test)

	return x_train, y_train, x_control_train, x_control_test, x_test, y_test

def checkNormalFit(x_train, y_train, x_control_train):
	train = []
	for i in range(0, len(y_train)):
		temp1 = np.append(x_train[i], y_train[i])
		temp2 = np.append(temp1, x_control_train[i])
		train.append(temp2)

	mean = np.mean(train, axis=0)
	cov = np.cov(train, rowvar=0)
	l = len(mean) - 2
	for i in range(0, l):
		for j in range(0, l):
			if i != j:
				cov[i][j] = 0

	for i in range(0, len(train[0])):
		data = []
		for elem in train:
			data.append(elem[i])

		def cdf(x):
			return st.norm.cdf(x, mean[i], math.sqrt(cov[i][i]))

		print(st.kstest(data, cdf))
