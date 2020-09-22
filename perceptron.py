# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 18:23:35 2020

@author: Leah N Busch
"""


#Write a program to train the Perceptron to learn the AND function starting from random weights.

#imports
import numpy as np
import matplotlib.pyplot as plt

#Data
x = [[0,0], [0,1],[1,0],[1,1]]
y	 = [0,0,0,1]
x_3d = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
y_3d = [0, 0, 0, 0, 0, 0, 0, 1]

class Perceptron():
	def __init__(self,x,y,learning_rate = 0.01):
		self.x = x
		self.y = y
		self.learning_rate = learning_rate
		self.w = self.initialize_weights()

	#initialize random weights
	def initialize_weights(self):
		try:
			return np.random.random_sample(np.shape(self.x)[1]+1)
		except:
			raise ValueError("Only one row of training data found")


	#predict one input
	def predict(self,x):

		try:
			prediction = np.dot(x, self.w[1:]) + self.w[0]
		except:
			raise ValueError("Wrong input data predictions")
		if prediction > 0:
			return 1
		else:
			return 0

	def calculate_error(self):
		error = 0
		for iterator, row in enumerate(self.x):
			error += self.y[iterator] - self.predict(row)
		return error

	def train(self):
		while self.calculate_error() != 0:
			for inputs, label in zip(self.x, self.y):
				prediction = self.predict(inputs)
				self.w[1:] += [self.learning_rate * (label - prediction)*i for i in inputs]
				self.w[0] += self.learning_rate * (label - prediction)


	def plot(self):
		#only runs if input shape 2
		try:
			if np.shape(self.x)[1] == 2:
				x_0 = np.asmatrix([self.x[counter] for counter,value in\
					    enumerate(self.y) if value == 0])
				x_1 = np.asmatrix([self.x[counter] for counter,value in \
					   enumerate(self.y) if value == 1])
				#line data
				x_boundary = np.linspace(0,1,100)
				slope = -(self.w[0] / self.w[2]) / (self.w[0] / self.w[1])
				y_intercept =  -self.w[0] / self.w[2]
				plt.plot(x_0[:,0],x_0[:,1],"o",x_1[:,0],x_1[:,1], "x",\
			 x_boundary,[y_intercept +slope * i for i in x_boundary])
				plt.xlabel("x1")
				plt.ylabel("x2")
				plt.xlim([-0.5,1.5])
				plt.ylim([-0.5,1.5])
				plt.title("Plot and Decision Boundary for 2 Inputs")
		except:
			raise ValueError("Cannot plot with dataset used")

def run_hw():
	print("#### Leah N Busch Homework 2 ####")
	p = Perceptron(x,y)
	p.train()
	p.plot()
	print("#### Weights for 2 input AND gate: [{:.2f},{:.2f},{:.2f}] ####".format(p.w[0],p.w[1],p.w[2]))
	print("#### Output for two input AND gate ####")
	print("####  x  |  y  |  o  ####")
	for row in x:
		print("####  {:d}  |  {:d}  |  {:d}  ####".format(row[0],row[1],p.predict(row)))

def run_3d():
	p = Perceptron(x_3d,y_3d)
	p.train()
	print("#### Weights for 3 input AND gate: [{:.2f},{:.2f},{:.2f},{:.2f}] ####".format(p.w[0],p.w[1],p.w[2],p.w[3]))
	print("#### Output for three input AND gate ####")
	print("####  x  |  y  |  z  |  o  ####")

	for row in x_3d:
		print("####  {:d}  |  {:d}  |  {:d}  |  {:d}  ####".format(row[0],row[1],row[2], p.predict(row)))


run_hw()
run_3d()