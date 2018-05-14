import game
import network
import random
import math
import copy
import numpy as np # debug
from itertools import chain

GENE_POOL_SIZE = 30

# value +/- (abs(value)*mutation) max value of the mutation variable
MUTATION_MAX_DEVIATION = 1.0
# Fraction of neural net's weights and biases to mutate
MUTATION_PORTION = 0.05

NETWORK_LAYER_SIZES = [len(game.board.board)*2, 30, len(game.board.board)]


class BotGenePool():

	def __init__(self, gene_pool_size):
		try:
			if (gene_pool_size <= 0):
				raise ValueError('Gene pool must have a positive size. Current size is {0}'.format(gene_pool_size))
		except ValueError as va:
			raise

		self.gene_pool = [network.Network(NETWORK_LAYER_SIZES) for b in range(gene_pool_size)]

	# Plays all bots in gene pool against all others and calculates fitness based on number of wins
	def play_gene_pool(self):
		# Fitness value
		win_count = [0 for i in range(len(self.gene_pool))]

		for i in range(len(self.gene_pool)):
			# Play with all other bots in gene pool
			for j in chain( range(i), range(i+1, len(self.gene_pool)) ):
				# Play bots
				try:
					winner = game.play_game([self.gene_pool[i] , self.gene_pool[j]])
				except IndexError as ie:
					print("Exactly two bots must play tic tac toe")
				game.board.clear()

				# Update win counts
				if winner == 1:		win_count[i] += 1
				elif winner == -1:	win_count[i] -= 1
				elif winner == 2:	win_count[j] += 1
				else : 				win_count[j] -= 1

		return win_count

	# Replaces the current gene pool with a new evolved one and returns it
	# <param> update_epoch:		number of epochs per which an update should be displayed.
	#								0 if no updates should be displayed 
	def evolve(self, epochs, update_epoch = 0):
		# Layout:
		#	50% top original gene pool
		#	50% mutated top original

		should_display = True

		for epoch in range(epochs):
			should_display = update_epoch != 0 and epoch % update_epoch == 0

			# original_portion = 0.5 + (0.5*epoch/epochs) # TODO
			original_portion = 0.5
			top_original_count = math.ceil(len(self.gene_pool) * original_portion)
			mutated_count = math.floor(len(self.gene_pool) * (1-original_portion))

			# get fitness
			fitness = self.play_gene_pool()

			# top original
			top_indeces = sorted( range(len(self.gene_pool)), key=lambda i: fitness[i], reverse=True )[:top_original_count]
			gene_pool = [self.gene_pool[i] for i in top_indeces]

			gene_pool += self.mutate(gene_pool, mutated_count)

			self.gene_pool = gene_pool
			# print(original_portion, gene_pool, top_original_count, mutated_count) # debug

			if should_display:
				print( 'epoch {0}:'.format(epoch))
				print( 'gene_pool biases: {0}'.format( sum([ sum([np.sum(b, axis=0) for b in genome.biases]) for genome in self.gene_pool ]) ) )
				print( 'fitness: {0}'.format(fitness) )
				print( 'top indeces: {0}, {1}'.format(top_indeces, [fitness[i] for i in top_indeces]) )
				print()

		return self.gene_pool


	# Mutates a gene pool and returns a mutated pool with a desired length
	# The number of mutated entries is such that a desired mutated pool length is achieved
	# <param> original_pool:		pool that should be mutated
	# <param> mutated_pool_length:	length of the mutated gene pool
	@staticmethod
	def mutate(original_pool, mutated_pool_length):

		# Ratio of original pool size to mutated pool size
		size_ratio = math.ceil(len(original_pool) / mutated_pool_length)
		mutated_pool = [None for i in range(mutated_pool_length)]

		j = 0
		for i in range(0, len(original_pool), size_ratio):
			mutated_pool[j] = BotGenePool.mutate_genome(original_pool[i])
			j += 1
			# print(original_pool[i].weights[0][:5]) # debug

		# Add a mutation if current number of mutations insufficient
		# TODO
		# Truncate if too many mutations
		# TODO

		return mutated_pool

	# Mutates a single genome
	# <param> genome:		neural network to mutate
	@staticmethod
	def mutate_genome(genome):
		# Get number of weights and biases to mutate
		mutate_weight_count = math.floor(genome.total_weights_count * MUTATION_PORTION)
		mutate_bias_count = math.floor(genome.total_biases_count * MUTATION_PORTION)

		# Get a number of random indeces of weights and biases to mutate
		mutate_weights_inds = [random.randint(0, genome.total_weights_count-1) for i in range(mutate_weight_count)]
		mutate_bias_inds = [random.randint(0, genome.total_biases_count-1) for i in range(mutate_bias_count)]

		# k - index in this layer
		# j - index in previous layer
		# Mutate all chosen weights and biases
		mutated_genome = copy.deepcopy(genome)
		for mut_w in mutate_weights_inds:
			layer, k, j = BotGenePool.weight_id_to_indeces(mut_w, mutated_genome)
			# print('mutating weights {0}/{1}/{2} = {3}'.format(layer,k,j, mutated_genome.weights[layer][k][j])) # debug
			mutated_genome.weights[layer][k][j] = BotGenePool.value_mutate(mutated_genome.weights[layer][k][j])
			# print('  mutated = {0}'.format(mutated_genome.weights[layer][k][j])) # debug

		for mut_b in mutate_bias_inds:
			layer, k = BotGenePool.bias_id_to_indeces(mut_b, mutated_genome)
			mutated_genome.biases[layer][k] = BotGenePool.value_mutate(mutated_genome.biases[layer][k])

		return mutated_genome

	# Mutates one value and returns the mutated value
	# Mutated value is between -MUTATION_MAX_DEVIATION*value <= value <= MUTATION_MAX_DEVIATION*value
	@staticmethod
	def value_mutate(value):
		# Mutation severity is uniform
		return value + math.fabs(value) * random.uniform(-MUTATION_MAX_DEVIATION, MUTATION_MAX_DEVIATION)

	# Returns coordinates of a weight based on an id and a neural net genome
	# <param> weight_id:	weight 1d id to be decoded into 3d coordinates
	@staticmethod
	def weight_id_to_indeces(weight_id, genome):
		
		# Find layer
		layer, weight_id = BotGenePool.find_layer(weight_id, [w.shape[0] * w.shape[1] for w in genome.weights])

		# Find index in this layer
		ind_in_this_layer = weight_id // genome.weights[layer].shape[1]
		weight_id -= ind_in_this_layer * genome.weights[layer].shape[1]

		# Find index in previous layer
		ind_in_prev_layer = weight_id

		return layer, ind_in_this_layer, ind_in_prev_layer

	@staticmethod
	def bias_id_to_indeces(bias_id, genome):
		# Find layer and index in this layer
		layer, ind_in_this_layer = BotGenePool.find_layer(bias_id, [b.shape[0] for b in genome.biases])

		return layer, ind_in_this_layer

	@staticmethod
	def find_layer(p_id, totals):
		layer = 0
		while p_id >= totals[layer]:
			p_id -= totals[layer]
			layer += 1
		return layer, p_id




bot_gene_pool = BotGenePool(GENE_POOL_SIZE)

smart_pool = bot_gene_pool.evolve(400, update_epoch = 20)

dumb_bot = smart_pool[1]
smart_bot = smart_pool[0]

game.play_game([smart_bot, dumb_bot], show_game = True)
