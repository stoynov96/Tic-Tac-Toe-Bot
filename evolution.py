import game
import network
import random
import math
import copy
import numpy as np
from itertools import chain

# standard deviation bounds for each mutation
MUTATION_DEVIATION_MIN = 2.0
MUTATION_DEVIATION_MAX = 10.0
# Fraction of neural net's weights and biases to mutate
MUTATION_PORTION = 0.80

# Minimum standard deviation of a pool's fitness values for it to not be considered convergent 
CONVERGENCE_SD = 1.0

# Fitness points added for a win
FIRST_WIN_AWARD = 5 # award for playing first and winning
SECOND_WIN_AWARD = 8 # award for playing second and winning
# Fitness points added for losing
FIRST_LOSE_AWARD = -8 # penalty for playing first and losing
SECOND_LOSE_AWARD = -5 # penalty for playing second and losing
# Fitness points added for a draw
FIRST_DRAW_AWARD = 0 # award for playing first and losing
SECOND_DRAW_AWARD = 2 # award for playing second and losing
# Fitness points added for being disqualified (placing a mark on a marked cell, etc.)
STUPID_AWARD = -8

NETWORK_LAYER_SIZES = [len(game.board.board)*2, 30, len(game.board.board)]


class Genome(object):

	def __init__(self, genome, fitness = 0, stupid = 0):
		self.genome = genome
		self.fitness = fitness
		self.stupid = stupid



class GenePool(object):

	# <param=pool> Genome objects that should make up the gene pool
	def __init__(self, pool):
		try:
			if (len(pool) < 1):
				raise ValueError('Gene pool must have a positive size. Current size is {0}'.format(len(gene_pool)))
		except ValueError as va:
			raise

		# Initialize gene pool
		self.pool = pool

		# Initialize top indeces based on fitness score (high to low)
		self.top_indeces = None

	# Detects if a gene pool is convergent and returns the result
	# <param=min_sd> minimum standard deviation required for the pool to not be considered convergent 
	def is_convergent(self, min_sd, should_display = False, indentation = ''):
		# Find fitness values standard deviation
		fitness_np_arr = np.array([ g.fitness for g in self.pool ])
		fitness_sd = np.std ( fitness_np_arr , axis = 0)
		# Display standard_deviation if needed
		if should_display:
			print(indentation + 'fitness array: {0}'.format(fitness_np_arr))
			print(indentation + 'fitness standard deviation: {0}'.format(fitness_sd))
		# Determine convergence
		return fitness_sd < min_sd


	# Add a list of genomes to the gene pool
	# <param=list_of_genomes> List of objects of type Genome to add to the gene pool
	def add_to_pool(self, list_of_genomes):
		if len(list_of_genomes) == 0: return
		try:
			if len(list_of_genomes) < 0:
				raise ValueError('list of genomes to add to a gene pool must be of a non-negative length')
			if not isinstance(list_of_genomes[0], Genome):
				raise TypeError('every element of the list of genomes to be added to gene pool must be of type Genome')
		except ValueError as ve:
			raise
		except TypeError as te:
			raise

		self.pool += list_of_genomes

	# Updates top indeces field and returns it
	def update_top_indeces(self, num_of_top_genomes):
		self.top_indeces = sorted( range(len(self.pool)), key=lambda i: self.pool[i].fitness, reverse=True )[:num_of_top_genomes]
		return self.top_indeces

	# Trims the gene pool so that only genomes with top fitness values are left sorted by their fitness value
	# The number of genomes left is determined by the number of entries in the top_indeces list
	# The remaining genomes are sorted by their fitness value
	def sort_and_trim(self):
		if not self.top_indeces or len(self.top_indeces) <= 0:
			print('WARNING! gene pool is trimmed based on a top indeces list that is void or doesn not exist')
		self.pool = [self.pool[i] for i in self.top_indeces]

	# Resets fitness and stupid values of all genomes in the gene pool
	def reset_fitness(self):
		for i in range(len(self.pool)):
			self.pool[i].fitness = 0
			self.pool[i].stupid = 0




class BotEvolution():

	def __init__(self, gene_pool_size, gene_pool_count):
		try:
			if (gene_pool_size <= 0):
				raise ValueError('Gene pool must have a positive size. Current size is {0}'.format(gene_pool_size))
		except ValueError as va:
			raise

		self.gene_pools = [ GenePool([ Genome(network.Network(NETWORK_LAYER_SIZES)) for b in range(gene_pool_size) ]) for i in range(gene_pool_count) ]
		self.fittest = []

	# Resets fitness and stupid values of all genomes in the gene pool
	def reset_fitness(self):
		for gene_pool in self.gene_pools:
			gene_pool.reset_fitness()


	# Plays all gene pools to determine their genomes' fitness
	def play_gene_pools(self):
		for gene_pool in self.gene_pools:
			# Play each gene pool
			self.play_gene_pool(gene_pool)


	# Plays all bots in gene pool against all others and calculates fitness based on number of wins
	def play_gene_pool(self, gene_pool):
		for i in range(len(gene_pool.pool)):
			# Play with all other bots in gene pool
			for j in chain( range(i), range(i+1, len(gene_pool.pool)) ):
				# Play bots
				try:
					winner = game.play_game([gene_pool.pool[i].genome , gene_pool.pool[j].genome])
				except IndexError as ie:
					print("Exactly two bots must play tic tac toe")
				game.board.clear() # TODO: Make sure this is safe to remove and remove it

				# Update win counts
				if winner == 1:
					gene_pool.pool[i].fitness += FIRST_WIN_AWARD
					gene_pool.pool[j].fitness += FIRST_LOSE_AWARD
				elif winner == -1:
					gene_pool.pool[i].fitness += STUPID_AWARD
					gene_pool.pool[i].stupid += 1

				elif winner == 2:
					gene_pool.pool[j].fitness += SECOND_WIN_AWARD
					gene_pool.pool[i].fitness += SECOND_LOSE_AWARD
				elif winner == -2: 				
					gene_pool.pool[j].fitness += STUPID_AWARD
					gene_pool.pool[j].stupid += 1
				elif winner == 0: # Draw
					gene_pool.pool[i].fitness += FIRST_DRAW_AWARD
					gene_pool.pool[j].fitness += SECOND_DRAW_AWARD


	# Replaces the current gene pool with a new evolved one and returns it
	# <param> update_generation:		number of generations per which an update should be displayed.
	#								0 if no updates should be displayed 
	def evolve(self, update_generation = 0):
		should_display = True
		generation = 0

		# Indeces of gene pools marked for deletion
		gp_to_delete = []

		while len(self.gene_pools) > 0:
			should_display = update_generation != 0 and generation % update_generation == 0

			# Gene pool content split
			original_count, mutated_count, bred_count, new_count = BotEvolution.get_pool_split([0.3, 0.6, 0.1], len(self.gene_pools[0].pool))

			# Get fitness for all gene pools
			# TODO: Display before and after. It could be that fitness values aren't updated (passed by value)
			self.reset_fitness()
			self.play_gene_pools()
			
			# Evolve all gene pools
			for i in range(len(self.gene_pools)):

				# Temporary gene pool to work with
				gp = self.gene_pools[i]

				# Top genomes to advance from last to this generation
				gp.update_top_indeces(original_count)				

				# Advance top genomes
				gp.sort_and_trim()

				# Check for convergence: TODO
				if gp.is_convergent(min_sd = 1.0):
					# Mark for deletion
					gp_to_delete.append(i)

					# Display state after the removal of gene pool
					should_display = True

					# Save the fittest genome
					self.fittest.append(gp.pool[0])

					# Notify of convergence
					print('Convergence detected @ i={0}...'.format(i))

					break

				# Breed genomes from different gene pools and add to this pool:
				min_fitness = gp.pool[-1].fitness # Smallest fitness value
				gp.add_to_pool( BotEvolution.breed_pools( self.gene_pools , bred_count , init_fitness = min_fitness) )

				# Mutate genomes and add to pool
				gp.add_to_pool( BotEvolution.mutate(gp, mutated_count, init_fitness = min_fitness) )

				# Insert new genomes into the pool
				gp.add_to_pool([network.Network(NETWORK_LAYER_SIZES) for i in range(new_count)])

				# Update gene pool with the temporary variable
				self.gene_pools[i] = gp

			# Display statistics
			if should_display:
				BotEvolution.display_stats(generation, self.gene_pools, 8)
				print()

			# Delete gene pools marked for deletion
			for i in range(len(gp_to_delete)-1, -1, -1):
				del self.gene_pools[i]
			gp_to_delete = []

			# Update generation counter
			generation += 1

		return self.fittest


	""" *----- Breeding -----* ===================================================================================================================================================="""

	# Breeds genomes from different gene pools and returns a single pool
	# <param=or_pools>		original gene pools from which genomes should be bred
	# <param=gp_length>		length of the new gene pool
	# <param=init_fitness>	initial fitness of newly created genomes
	# <return> bred list of objects of type Genome with a specified length
	@staticmethod
	def breed_pools(or_pools, gp_length, init_fitness = 0):
		# Make sure none of the pools is empty
		try:
			for gene_pool in or_pools:
				if len(gene_pool.pool) < 1:
					raise ValueError('Breeding gene pools must both be of positive length')
		except ValueError as va:
			raise
		except TypeError as te:
			raise TypeError('The original gene pools must be of type GenePool')

		# Create random combinations of genomes from the gene pools to breed
		genome_combo_indeces = [ [ random.randint(0,len(gene_pool.pool)-1) for gene_pool in or_pools ] for i in range(gp_length) ]
			# Clarification example: if genome_combo_indeces[3] = [1,4,17] that means
			# For genome [3] of the bred gene pool, take genome [1] from original gene pool [0],
			# genome [4] from original gene pool [1] and genome [17] from original gene pool [2]

		# combo = [5,2,0] => [ or[0][5], or[1][2], or[2][0] ]
		# Return a new gene pool with a length gp_length
		return [ BotEvolution.breed_n( [ or_pools[i].pool[c] for i,c in zip(range(len(or_pools)), combo) ] , init_fitness = init_fitness ) for combo in genome_combo_indeces ]

	# Breeds given bots and returns a new bot as a result 
	# <param=bots>	list<Genome> list of bots to breed
	# <param=init_fitness> initial fitness of the newly created bot
	# <return>		new bot that was the result of the breeding
	@staticmethod
	def breed_n(bots, init_fitness):
		# Full neuron selection
		# Initialize the new bots
		new_bot = Genome(network.Network(bots[0].genome.layers), fitness = init_fitness, stupid=111) # specific stupid value for display purposes

		# Breed bots
		for layer in range(len(bots[0].genome.weights)):
			for neuron in range(len(bots[0].genome.weights[layer])):
				parent_gene = random.randint(0,len(bots)-1)
				new_bot.genome.weights[layer][neuron] = np.copy(bots[parent_gene].genome.weights[layer][neuron])
				new_bot.genome.biases[layer][neuron] = np.copy(bots[parent_gene].genome.biases[layer][neuron])

		# Return the new bots
		return new_bot





	""" *---- Mutations ----* ===================================================================================================================================================="""


	# Mutates a gene pool and returns a mutated pool with a desired length
	# The number of mutated entries is such that a desired mutated pool length is achieved
	# <param=original_pool>			pool that should be mutated
	# <param=mutated_pool_length>	length of the mutated gene pool
	# <param=init_fitness> initial fitness of newly created genomes by mutation
	# <return> list of Genome objects containing the mutated gene pool
	@staticmethod
	def mutate(original_pool, mutated_pool_length, init_fitness):
		# Type checking
		try:
			if not isinstance(original_pool, GenePool):
				raise TypeError('Gene pool to mutate must be of type GenePool')
		except TypeError as te:
			raise

		# Ratio of original pool size to mutated pool size
		mutated_pool = [None for i in range(mutated_pool_length)]

		# Get random indeces of original pool to mutate
		mutated_indeces = [ random.randint(0, len(original_pool.pool)-1) for i in range(mutated_pool_length) ]

		# Calculate maximum and minimum fitness
		max_fitness = original_pool.pool[0].fitness
		min_fitness = original_pool.pool[-1].fitness

		j = 0
		for i in mutated_indeces:
			# Calculate standard deviation for each genome mutation operation
			normal_dist_to_max = (max_fitness-original_pool.pool[i].fitness) / (max_fitness+1 - min_fitness) # prevent division by zero
			mutation_deviation = MUTATION_DEVIATION_MIN + normal_dist_to_max * (MUTATION_DEVIATION_MAX - MUTATION_DEVIATION_MIN)

			# Mutate genome
			mutated_pool[j] = BotEvolution.mutate_genome(original_pool.pool[i], mutation_deviation, init_fitness)
			j += 1

		return mutated_pool

	# Mutates a single genome
	# <param=genome>		genome of type Genome containing a neural net to mutate
	# <param=mutation_deviation>	standard deviation for each mutation
	# <param=init_fitness> initial fitness value of newly mutated genomes
	# <return> Genome - the genomoe that was the result of the mutation
	@staticmethod
	def mutate_genome(genome, mutation_deviation, init_fitness):
		mutated_genome = copy.deepcopy(genome.genome)
		# Get number of weights and biases to mutate
		mutate_weight_count = math.floor(mutated_genome.total_weights_count * MUTATION_PORTION)
		mutate_bias_count = math.floor(mutated_genome.total_biases_count * MUTATION_PORTION)

		# Get a number of random indeces of weights and biases to mutate
		mutate_weights_inds = [random.randint(0, mutated_genome.total_weights_count-1) for i in range(mutate_weight_count)]
		mutate_bias_inds = [random.randint(0, mutated_genome.total_biases_count-1) for i in range(mutate_bias_count)]

		# k - index in this layer
		# j - index in previous layer
		# Mutate all chosen weights and biases
		for mut_w in mutate_weights_inds:
			layer, k, j = BotEvolution.weight_id_to_indeces(mut_w, mutated_genome)
			mutated_genome.weights[layer][k][j] = BotEvolution.value_mutate(mutated_genome.weights[layer][k][j], mutation_deviation)

		for mut_b in mutate_bias_inds:
			layer, k = BotEvolution.bias_id_to_indeces(mut_b, mutated_genome)
			mutated_genome.biases[layer][k] = BotEvolution.value_mutate(mutated_genome.biases[layer][k], mutation_deviation)

		return Genome(mutated_genome, fitness = init_fitness, stupid = 222) # specific stupid value for display purposes

	# Mutates one value and returns the mutated value
	# Changes the mutated value by replacing it with a normal random variable with a mean the old value
	# and a specified standard deviation
	@staticmethod
	def value_mutate(value, standard_deviation):
		# Mutation severity is uniform
		# return value + math.fabs(value) * random.uniform(-MUTATION_MAX_DEVIATION, MUTATION_MAX_DEVIATION) # obsolete
		# return np.random.normal(loc = value, scale = abs(standard_deviation * value))
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

	# Displays statistics about a generation
	# <param=list_limit>	maximum number of elements to display per list
	@staticmethod
	def display_stats(generation, gene_pools, list_limit):
		print( 'generation {0}:'.format(generation))
		for i in range(len(gene_pools)):
			gene_pool = gene_pools[i]
			print( '  Gene Pool {0}:'.format(i))
			# print( '    gene_pool biases: {0}'.format( sum([ sum([np.sum(b, axis=0) for b in genome.biases]) for genome in gene_pool ]) ) )
			print( '    gene_pool avg biases: {0}'.format( sum([genome.genome.get_average_bias() for genome in gene_pool.pool])/len(gene_pool.pool) ) )
			print( '    gene_pool size: {0}'.format( len(gene_pool.pool) ) )
			print( '    top stupidity/fitness: {0} / {1}'.format([genome.stupid for genome in gene_pool.pool[:list_limit]],
																 [genome.fitness for genome in gene_pool.pool[:list_limit]]) )

			# Print out the board after a game of the top two bots
			game.play_game([g.genome for g in gene_pool.pool[0:2]])
			game.board.display(indentation = '    ')