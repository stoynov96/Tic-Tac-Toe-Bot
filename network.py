import numpy as np

class Network(object):
	def __init__(self, layers):
		self.layer_count = len(layers)
		self.layers = layers
		self.biases = [np.random.randn(l,1) for l in layers[1:]]
		self.weights = [np.random.randn(l, prev_l) for prev_l,l in zip(layers[:-1],layers[1:])]

	# Feeds an input layer vector through the network
	# <return> Output layer of the given input
	def feedforward(self, activation):
		for w,b in zip(self.weights, self.biases):
			activation = sigmoid( np.dot(w, activation) + b )
			# print('activation:  {0}\n    weights: {1}\n    biases:  {2}'.format(activation.shape, w.shape, b.shape))

		return activation


def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

