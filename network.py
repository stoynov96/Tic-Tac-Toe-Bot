import numpy as np

class Network(object):
	# TODO: add a parameter to allow for initializing weights and biases to zero to reduce overhead
	# for when they will be changed immideately
	def __init__(self, layers):
		self.layer_count = len(layers)
		self.layers = layers
		self.biases = [np.random.randn(l,1) for l in layers[1:]]
		self.weights = [np.random.randn(l, prev_l) for prev_l,l in zip(layers[:-1],layers[1:])]

		# needed to optimize evolution
		self.total_weights_count , self.total_biases_count = self.get_total_parameters_count()

	# Feeds an input layer vector through the network
	# <return> Output layer of the given input
	def feedforward(self, activation):
		for w,b in zip(self.weights, self.biases):
			activation = sigmoid( np.dot(w, activation) + b )
			# print('activation:  {0}\n    weights: {1}\n    biases:  {2}'.format(activation.shape, w.shape, b.shape))

		return activation

	# Returns the total number of weights and biases in the neural net
	def get_total_parameters_count(self):
		total_weights = 0
		total_biases = 0

		for w in self.weights:
			total_weights += w.shape[0] * w.shape[1]
		for b in self.biases:
			total_biases += b.shape[0]

		return total_weights, total_biases

	# Returns the average bias of the neural network
	def get_average_bias(self):
		sum_aves = sum ([ np.average(b)*b.shape[0] for b in self.biases ])
		return sum_aves / sum( [b.shape[0] for b in self.biases] )



def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

