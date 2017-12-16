import numpy as np
import matplotlib.pyplot as plt
# import scipy

class net():
	def __init__(self, learning_rate=10, mem_size=50, reg=0.00001):
		# NN architecture definition
		self.input_size = 64*48 # 3072
		self.hidden1n = 1000
		self.hidden2n = 700
		self.hidden3n = 400
		self.output_size = 200
		# transition weights
		self.W1 = np.random.normal(scale=1, size=(self.hidden1n, self.input_size))
		self.W2 = np.random.normal(scale=1, size=(self.hidden2n, self.hidden1n))
		self.W3 = np.random.normal(scale=1, size=(self.hidden3n, self.hidden2n))
		self.W4 = np.random.normal(scale=1, size=(self.output_size, self.hidden3n))
		# transition offsets
		self.b1 = np.zeros(self.hidden1n)
		self.b2 = np.zeros(self.hidden2n)
		self.b3 = np.zeros(self.hidden3n)
		self.b4 = np.zeros(self.output_size)
		# hyper parameters
		self.learning_rate = learning_rate
		self.reg = reg
		# batch memory 
		self.mem_size = mem_size
		self.memory = np.zeros((self.mem_size, self.input_size))
		self.mem_labels = np.zeros((self.mem_size, self.output_size))
		self.mem_i = 0

	def softmax(self, x):
		shifted_input = x - np.max(x, axis=0)
		numerators = np.exp(shifted_input)
		return (numerators / np.sum(numerators, axis=0))

	def sigmoid(self, x):
		s = 1 / (1 + np.exp(-x))
		return s
		# return expit(x)

	def forward_prop(self, data, labels):
		a0 = data.T

		z1 = self.W1 @ a0 + np.tile(self.b1, (self.mem_size,1)).T
		a1 = self.sigmoid(z1)

		z2 = self.W2 @ a1 + np.tile(self.b2, (self.mem_size,1)).T
		a2 = self.sigmoid(z2)

		z3 = self.W3 @ a2 + np.tile(self.b3, (self.mem_size,1)).T
		a3 = self.sigmoid(z3)

		z4 = self.W4 @ a3 + np.tile(self.b4, (self.mem_size,1)).T
		a4 = self.softmax(z4)

		cost = -np.sum(labels.T * np.log(a4), axis=0)

		return a1, a2, a3, a4, cost

	def backward_prop(self, data, labels):
		#  Perform a single forward pass
		a1, a2, a3, a4, cost = self.forward_prop(data, labels)
		#  Calculate the layer deltas
		d4 = a4-labels.T
		d3 = self.W4.T @ d4 * a3 * (1 - a3)
		d2 = self.W3.T @ d3 * a2 * (1 - a2)
		d1 = self.W2.T @ d2 * a1 * (1 - a1)

		(n,m) = data.shape
		dW1 = d1 @ data / m
		dW2 = d2 @ a1.T / m
		dW3 = d3 @ a2.T / m
		dW4 = d4 @ a3.T / m
		db1 = np.sum(d1, axis=1).reshape(self.b1.shape) / m
		db2 = np.sum(d2, axis=1).reshape(self.b2.shape) / m
		db3 = np.sum(d3, axis=1).reshape(self.b3.shape) / m
		db4 = np.sum(d4, axis=1).reshape(self.b4.shape) / m

		return dW1, dW2, dW3, dW4, db1, db2, db3, db4

	def train_memory(self):
		(m, n) = self.memory.shape
		dW1, dW2, dW3, dW4, db1, db2, db3, db4 = self.backward_prop(self.memory, self.mem_labels)

		self.W1 = (1-self.learning_rate*self.reg)*self.W1 - self.learning_rate * dW1
		self.W2 = (1-self.learning_rate*self.reg)*self.W2 - self.learning_rate * dW2
		self.W3 = (1-self.learning_rate*self.reg)*self.W3 - self.learning_rate * dW3
		self.W4 = (1-self.learning_rate*self.reg)*self.W4 - self.learning_rate * dW4
		self.b1 = self.b1 - self.learning_rate * db1
		self.b2 = self.b2 - self.learning_rate * db2
		self.b3 = self.b3 - self.learning_rate * db3
		self.b4 = self.b4 - self.learning_rate * db4

	def memorize(self, img_vec, label):
		self.memory[self.mem_i] = img_vec
		self.mem_labels[self.mem_i] = label
		self.mem_i = (self.mem_i + 1) % self.mem_size

	def predict(self, img_vec):
		a0 = img_vec

		z1 = self.W1 @ a0 + self.b1
		a1 = self.sigmoid(z1)

		z2 = self.W2 @ a1 + self.b2
		a2 = self.sigmoid(z2)

		z3 = self.W3 @ a2 + self.b3
		a3 = self.sigmoid(z3)

		z4 = self.W4 @ a3 + self.b4
		a4 = self.softmax(z4)

		return np.argmax(a4, axis=0)

	def save_params(self, filename):
		with open(filename, 'wb') as outfile:
			np.savez(outfile, W1=self.W1, W2=self.W2, W3=self.W3, W4=self.W4,\
					b1=self.b1, b2=self.b2, b3=self.b3, b4=self.b4)

	def load_params(self, filename):
		data = np.load(filename)
		self.W1 = data['W1']
		self.W2 = data['W2']
		self.W3 = data['W3']
		self.W4 = data['W4']
		self.b1 = data['b1']
		self.b2 = data['b2']
		self.b3 = data['b3']
		self.b4 = data['b4']










