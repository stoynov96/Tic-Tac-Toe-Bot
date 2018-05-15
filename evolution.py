import game
import network
import random
import math
import copy
import numpy as np
import json
from itertools import chain

GENE_POOL_SIZE = 50

# standard deviation bounds for each mutation
MUTATION_DEVIATION_MIN = 0.5
MUTATION_DEVIATION_MAX = 10.0
# Fraction of neural net's weights and biases to mutate
MUTATION_PORTION = 0.05

# Fitness points awarded for a win
WIN_AWARD = 5
# Fitness points deducted for losing
LOSE_PENALTY = 2
# Fitness points deducted for being disqualified (placing a mark on a marked cell, etc.)
STUPID_PENALTY = 10

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
		fitness = [0 for i in range(len(self.gene_pool))]
		stupid_count = [0 for i in fitness]

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
				if winner == 1:
					fitness[i] += WIN_AWARD
					fitness[j] -= LOSE_PENALTY
				elif winner == -1:
					fitness[i] -= STUPID_PENALTY
					stupid_count[i] += 1

				elif winner == 2:
					fitness[j] += WIN_AWARD
					fitness[i] -= LOSE_PENALTY
				elif winner == -2: 				
					fitness[j] -= STUPID_PENALTY
					stupid_count[j] += 1

		return fitness, stupid_count

	# Replaces the current gene pool with a new evolved one and returns it
	# <param> update_epoch:		number of epochs per which an update should be displayed.
	#								0 if no updates should be displayed 
	def evolve(self, epochs, update_epoch = 0):
		should_display = True

		for epoch in range(epochs):
			should_display = update_epoch != 0 and epoch % update_epoch == 0

			# Gene pool content split
			original_count, mutated_count, bred_count, new_count = BotGenePool.get_pool_split([0.3, 0.3, 0.1], len(self.gene_pool))

			# Get fitness
			fitness, stupid = self.play_gene_pool()

			# Top original
			top_indeces = sorted( range(len(self.gene_pool)), key=lambda i: fitness[i], reverse=True )[:original_count]
			gene_pool = [self.gene_pool[i] for i in top_indeces]

			# Combine genomes
			combined_indeces = random.sample(range(0,len(gene_pool)), bred_count)
			gene_pool += BotGenePool.combine([gene_pool[i] for i in combined_indeces])

			# Mutate genomes
			mutation_fitness = [fitness[i] for i in top_indeces] + [ fitness[top_indeces[len(top_indeces)-1]] for i in range(mutated_count)]
				# assigning the minimum fitness to the bred genomes to encourage their mutation
			gene_pool += BotGenePool.mutate(gene_pool, mutated_count, mutation_fitness)

			# Insert new genomes
			gene_pool += [network.Network(NETWORK_LAYER_SIZES) for i in range(new_count)]

			self.gene_pool = gene_pool

			# Display statistics
			if should_display:
				print( 'epoch {0}:'.format(epoch))
				print( 'gene_pool biases: {0}'.format( sum([ sum([np.sum(b, axis=0) for b in genome.biases]) for genome in self.gene_pool ]) ) )
				print( 'gene_pool size: {0}'.format( len(gene_pool) ) )
				# print( 'fitness: {0}'.format(fitness) )
				# print( 'stupidity/fitness:     {0} / {1}'.format(stupid, fitness) )
				print( 'top stupidity/fitness: {0} / {1}'.format([stupid[i] for i in top_indeces[:10]], [fitness[i] for i in top_indeces[:10]]) )
				print()

		return self.gene_pool


	# Combines n genomes and returns b decendents
	# <param> original_pool:	gene pool from which genomes should be combined
	# <returns> combined gene pool with the length of the original. None of the original genomes were preserved
	@staticmethod
	def combine(original_pool):
		# Combine the n genomes
		gene_pool = [network.Network(genome.layers) for genome in original_pool]

		layer_count = len(original_pool[0].layers) - 1
		for i in range( len(gene_pool) ):
			for j in range( layer_count ):
				gene_pool[i].weights[j] = np.copy( original_pool[ (i+j) % len(gene_pool) ].weights[ j ] )
				gene_pool[i].biases[j] =  np.copy( original_pool[ (i+j) % len(gene_pool) ].biases [ j ] )


		# Mutate the resulting genomes
		return gene_pool


	# Mutates a gene pool and returns a mutated pool with a desired length
	# The number of mutated entries is such that a desired mutated pool length is achieved
	# <param> original_pool:		pool that should be mutated
	# <param> mutated_pool_length:	length of the mutated gene pool
	# <param> original_fitness:		sorted fitness of original genomes (descending)
	@staticmethod
	def mutate(original_pool, mutated_pool_length, original_fitness):

		# Ratio of original pool size to mutated pool size
		mutated_pool = [None for i in range(mutated_pool_length)]

		# Get random indeces of original pool to mutate
		mutated_indeces = random.sample(range(0,len(original_pool)), mutated_pool_length)

		# Calculate maximum and minimum fitness
		max_fitness = original_fitness[0]
		min_fitness = original_fitness[len(original_fitness)-1]

		j = 0
		# print('mut pool: {0}'.format(original_fitness) ) # debug
		for i in mutated_indeces:
			# Calculate standard deviation for each genome mutation operation
			normal_dist_to_max = (max_fitness-original_fitness[i]) / (max_fitness+1 - min_fitness) # prevent division by zero
			mutation_deviation = MUTATION_DEVIATION_MIN + normal_dist_to_max * (MUTATION_DEVIATION_MAX - MUTATION_DEVIATION_MIN)
			# print('  i={0} ft={1} md={2}'.format(i, original_fitness[i],mutation_deviation) ) # debug

			# Mutate genome
			mutated_pool[j] = BotGenePool.mutate_genome(original_pool[i], mutation_deviation)
			j += 1

		return mutated_pool

	# Mutates a single genome
	# <param> genome:		neural network to mutate
	# <param> mutation_deviation:	standard deviation for each mutation
	@staticmethod
	def mutate_genome(genome, mutation_deviation):
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
			mutated_genome.weights[layer][k][j] = BotGenePool.value_mutate(mutated_genome.weights[layer][k][j], mutation_deviation)
			# print('  mutated = {0}'.format(mutated_genome.weights[layer][k][j])) # debug

		for mut_b in mutate_bias_inds:
			layer, k = BotGenePool.bias_id_to_indeces(mut_b, mutated_genome)
			mutated_genome.biases[layer][k] = BotGenePool.value_mutate(mutated_genome.biases[layer][k], mutation_deviation)

		return mutated_genome

	# Mutates one value and returns the mutated value
	# Changes the mutated value by replacing it with a normal random variable with a mean the old value
	# and a specified standard deviation
	@staticmethod
	def value_mutate(value, standard_deviation):
		# Mutation severity is uniform
		# return value + math.fabs(value) * random.uniform(-MUTATION_MAX_DEVIATION, MUTATION_MAX_DEVIATION) # obsolete
		return np.random.normal(loc = value, scale = standard_deviation)

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

	# Returns individual sizes of genome splits
	# Example if fractions = [0.8,0.1] and pool_size = 100, this returns 80,10,10
	# (last fraction is not needed to be passed. It is assumed tha it completes to 1)
	@staticmethod
	def get_pool_split(fractions, pool_size):

		total_fractions = sum(fractions)
		# Check if input is meaningful
		try:
			if total_fractions > 1:
				raise ValueError('Gene pool fractions must not be sum up to more than 1.0')
		except ValueError as va:
			raise

		# Split
		split = [f * pool_size for f in fractions]
		split.append( pool_size - total_fractions*pool_size )

		# Make integers
		split = [int(round(s)) for s in split]

		return split





bot_gene_pool = BotGenePool(GENE_POOL_SIZE)

smart_pool = bot_gene_pool.evolve(1001, update_epoch = 20)

cont = 'c'
while (cont == 'c'):

	ind_sb = int(input("Smart Bot: "))
	ind_db = int(input("Dumb Bot: "))

	dumb_bot = smart_pool[ind_db]
	smart_bot = smart_pool[ind_sb]

	game.play_game([smart_bot, dumb_bot], show_game = True)
	cont = input("Game Over. Enter c to continue: ")

# Export winner network
biases = [ [list(b) for b in biases_layer] for biases_layer in smart_bot.biases ]
weights = [ [list(w) for w in weights_layer] for weights_layer in smart_bot.weights ]
nNet = {'biases': biases, 'weights': weights}

with open ('tictactoe_net_parameters.json', 'w') as outfile:
	json.dump(nNet, outfile)
