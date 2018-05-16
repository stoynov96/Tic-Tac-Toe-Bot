import game
import network
import random
import math
import copy
import numpy as np
from itertools import chain

# standard deviation bounds for each mutation
MUTATION_DEVIATION_MIN = 0.5
MUTATION_DEVIATION_MAX = 10.0
# Fraction of neural net's weights and biases to mutate
MUTATION_PORTION = 0.05

# Fitness points added for a win
FIRST_WIN_AWARD = 5 # award for playing first and winning
SECOND_WIN_AWARD = 8 # award for playing second and winning
# Fitness points added for losing
FIRST_LOSE_AWARD = -5 # penalty for playing first and losing
SECOND_LOSE_AWARD = -2 # penalty for playing second and losing
# Fitness points added for a draw
FIRST_DRAW_AWARD = -1 # award for playing first and losing
SECOND_DRAW_AWARD = 0 # award for playing second and losing
# Fitness points added for being disqualified (placing a mark on a marked cell, etc.)
STUPID_AWARD = -12

NETWORK_LAYER_SIZES = [len(game.board.board)*2, 30, len(game.board.board)]


class BotEvolution():

	def __init__(self, gene_pool_size, gene_pool_count):
		try:
			if (gene_pool_size <= 0):
				raise ValueError('Gene pool must have a positive size. Current size is {0}'.format(gene_pool_size))
		except ValueError as va:
			raise

		self.gene_pools = [ [network.Network(NETWORK_LAYER_SIZES) for b in range(gene_pool_size)] for i in range(gene_pool_count) ]

	# Plays all gene pools to determine their genomes' fitness
	def play_gene_pools(self):
		# TODO: optimize
		fitness, stupid = [], []
		for gene_pool in self.gene_pools:
			# Play each gene pool
			f,s = self.play_gene_pool(gene_pool)

			# Record fitness and stupid for each
			fitness.append(f)
			stupid.append(s)

		return fitness, stupid


	# Plays all bots in gene pool against all others and calculates fitness based on number of wins
	def play_gene_pool(self, gene_pool):
		# Fitness value
		fitness = [0 for i in range(len(gene_pool))]
		stupid_count = [0 for i in fitness]

		for i in range(len(gene_pool)):
			# Play with all other bots in gene pool
			for j in chain( range(i), range(i+1, len(gene_pool)) ):
				# Play bots
				try:
					winner = game.play_game([gene_pool[i] , gene_pool[j]])
				except IndexError as ie:
					print("Exactly two bots must play tic tac toe")
				game.board.clear() # TODO: Make sure this is safe to remove and remove it

				# Update win counts
				if winner == 1:
					fitness[i] += FIRST_WIN_AWARD
					fitness[j] += FIRST_LOSE_AWARD
				elif winner == -1:
					fitness[i] += STUPID_AWARD
					stupid_count[i] += 1

				elif winner == 2:
					fitness[j] += SECOND_WIN_AWARD
					fitness[i] += SECOND_LOSE_AWARD
				elif winner == -2: 				
					fitness[j] += STUPID_AWARD
					stupid_count[j] += 1
				elif winner == 0: # Draw
					fitness[i] += FIRST_DRAW_AWARD
					fitness[j] += SECOND_DRAW_AWARD

		return fitness, stupid_count

	# Replaces the current gene pool with a new evolved one and returns it
	# <param> update_epoch:		number of epochs per which an update should be displayed.
	#								0 if no updates should be displayed 
	def evolve(self, epochs, update_epoch = 0):
		should_display = True

		for epoch in range(epochs):
			should_display = update_epoch != 0 and epoch % update_epoch == 0

			# Gene pool content split
			original_count, mutated_count, bred_count, new_count = BotEvolution.get_pool_split([0.3, 0.3, 0.1], len(self.gene_pools[0]))

			# Get fitness for all gene pools
			fitness_list, stupid_list = self.play_gene_pools()

			sorted_ind_lists = []
			
			# Evolve all gene pools
			iter_zip = zip(range(len(self.gene_pools)), fitness_list, stupid_list)
			for i, fitness, stupid in iter_zip:

				# Temporary gene pool to work with
				gene_pool = self.gene_pools[i]

				# Top genomes to advance from last to this epoch
				top_indeces = sorted( range(len(gene_pool)), key=lambda i: fitness[i], reverse=True )[:original_count]
				sorted_ind_lists.append(top_indeces)
				# Advance top genomes
				gene_pool = [gene_pool[i] for i in top_indeces]

				# Breed genomes from different gene pools and add to this pool: TODO
				# bred_indeces = random.sample(range(0,len(gene_pool)), bred_count)
				gene_pool += BotEvolution.breed_pools( self.gene_pools , bred_count )

				# Mutate genomes and add to pool
				mutation_fitness = [fitness[i] for i in top_indeces] + [ fitness[top_indeces[len(top_indeces)-1]] for i in range(bred_count)]
					# assigning the minimum fitness to the bred genomes to encourage their mutation
				gene_pool += BotEvolution.mutate(gene_pool, mutated_count, mutation_fitness)

				# Insert new genomes into the pool
				gene_pool += [network.Network(NETWORK_LAYER_SIZES) for i in range(new_count)]

				# Update gene pool with the temporary variable
				self.gene_pools[i] = gene_pool

			# Display statistics
			if should_display:
				BotEvolution.display_stats(epoch, self.gene_pools, sorted_ind_lists, fitness_list, stupid_list, 8)

		return self.gene_pools


	""" *----- Breeding -----* ===================================================================================================================================================="""

	# Breeds genomes from different gene pools and returns a single pool
	# <param=or_pools>		original gene pools from which genomes should be bred
	# <param=gp_length>		length of the new gene pool
	# <return> bred gene pool with a specified length
	@staticmethod
	def breed_pools(or_pools, gp_length):
		# Make sure none of the pools is empty
		try:
			for pool in or_pools:
				if len(pool) < 1:
					raise ValueError('Breeding gene pools must both be of positive length')
		except ValueError as va:
			raise

		# Create random combinations of genomes from the gene pools to breed
		genome_combo_indeces = [ [ random.randint(0,len(pool)-1) for pool in or_pools ] for i in range(gp_length) ]
			# Clarification example: if genome_combo_indeces[3] = [1,4,17] that means
			# For genome [3] of the bred gene pool, take genome [1] from original gene pool [0],
			# genome [4] from original gene pool [1] and genome [17] from original gene pool [2]
		# print(genome_combo_indeces) # debug

		# combo = [5,2,0] => [ or[0][5], or[1][2], or[2][0] ]
		# Return a new gene pool with a length equal to the combined length of the original two
		return [ BotEvolution.breed_n( [ or_pools[i][c] for i,c in zip(range(len(or_pools)), combo) ] ) for combo in genome_combo_indeces ]

	# Breeds given bots and returns a new bot as a result 
	# <param> bots:	pool of bots to breed
	# <return>		new bot that was the result of the breeding
	@staticmethod
	def breed_n(bots):
		# Full neuron selection
		# Initialize the new bots
		new_bot = network.Network(bots[0].layers)

		# Breed bots
		for layer in range(len(bots[0].weights)):
			for neuron in range(len(bots[0].weights[layer])):
				parent_gene = random.randint(0,len(bots)-1)
				new_bot.weights[layer][neuron] = np.copy(bots[parent_gene].weights[layer][neuron])
				new_bot.biases[layer][neuron] = np.copy(bots[parent_gene].biases[layer][neuron])

		# Return the new bots
		return new_bot





	""" *---- Mutations ----* ===================================================================================================================================================="""


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
			mutated_pool[j] = BotEvolution.mutate_genome(original_pool[i], mutation_deviation)
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
			layer, k, j = BotEvolution.weight_id_to_indeces(mut_w, mutated_genome)
			# print('mutating weights {0}/{1}/{2} = {3}'.format(layer,k,j, mutated_genome.weights[layer][k][j])) # debug
			mutated_genome.weights[layer][k][j] = BotEvolution.value_mutate(mutated_genome.weights[layer][k][j], mutation_deviation)
			# print('  mutated = {0}'.format(mutated_genome.weights[layer][k][j])) # debug

		for mut_b in mutate_bias_inds:
			layer, k = BotEvolution.bias_id_to_indeces(mut_b, mutated_genome)
			mutated_genome.biases[layer][k] = BotEvolution.value_mutate(mutated_genome.biases[layer][k], mutation_deviation)

		return mutated_genome

	# Mutates one value and returns the mutated value
	# Changes the mutated value by replacing it with a normal random variable with a mean the old value
	# and a specified standard deviation
	@staticmethod
	def value_mutate(value, standard_deviation):
		# Mutation severity is uniform
		# return value + math.fabs(value) * random.uniform(-MUTATION_MAX_DEVIATION, MUTATION_MAX_DEVIATION) # obsolete
		return np.random.normal(loc = value, scale = standard_deviation)


	""" *-- Miscellaneous --* ===================================================================================================================================================="""

	# Returns coordinates of a weight based on an id and a neural net genome
	# <param> weight_id:	weight 1d id to be decoded into 3d coordinates
	@staticmethod
	def weight_id_to_indeces(weight_id, genome):
		
		# Find layer
		layer, weight_id = BotEvolution.find_layer(weight_id, [w.shape[0] * w.shape[1] for w in genome.weights])

		# Find index in this layer
		ind_in_this_layer = weight_id // genome.weights[layer].shape[1]
		weight_id -= ind_in_this_layer * genome.weights[layer].shape[1]

		# Find index in previous layer
		ind_in_prev_layer = weight_id

		return layer, ind_in_this_layer, ind_in_prev_layer

	@staticmethod
	def bias_id_to_indeces(bias_id, genome):
		# Find layer and index in this layer
		layer, ind_in_this_layer = BotEvolution.find_layer(bias_id, [b.shape[0] for b in genome.biases])

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

	# Displays statistics about an epoch
	# <param> top_indeces:	gene pool indeces sorted in descending order based on fitness for each gene pool
	# <param> fitness_list:	fitness values of each genome in all genome pools
	# <param> stupid_list:	number of disqualifying moves each genome has made for all genome pools
	# <param> list_limit:	maximum number of elements to display per list
	@staticmethod
	def display_stats(epoch, gene_pools, sorted_ind_lists, fitness_list, stupid_list, list_limit):
		print( 'epoch {0}:'.format(epoch))
		for gene_pool, fitness, stupid, top_indeces in zip(gene_pools, fitness_list, stupid_list, sorted_ind_lists):
			print( '  Gene Pool:')
			print( '    gene_pool biases: {0}'.format( sum([ sum([np.sum(b, axis=0) for b in genome.biases]) for genome in gene_pool ]) ) )
			print( '    gene_pool size: {0}'.format( len(gene_pool) ) )
			print( '    top stupidity/fitness: {0} / {1}'.format([stupid[i] for i in top_indeces[:list_limit]], [fitness[i] for i in top_indeces[:list_limit]]) )

			# Print out the board after a game of the top two bots
			game.play_game(gene_pool[0:2])
			game.board.display(indentation = '    ')

		print()
