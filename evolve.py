import evolution
import game
import json



GENE_POOL_COUNT = 2
GENE_POOL_SIZE = 40

bot_gene_pool = evolution.BotEvolution(GENE_POOL_SIZE, GENE_POOL_COUNT)


cont = 'c'
while (cont == 'c'):
	smart_pools = bot_gene_pool.evolve(int(input('Generations to evolve for...')), update_generation = 20)

	gp_index = int(input("Gene pool to play: "))

	ind_sb = int(input("Smart Bot: "))
	ind_db = int(input("Dumb Bot: "))

	try:
		dumb_bot = smart_pools[gp_index].pool[ind_db].genome
		smart_bot = smart_pools[gp_index].pool[ind_sb].genome
	except IndexError as ie:
		print("Invalid gene pool index. Please choose a valid gene pool")
		continue


	game.play_game([smart_bot, dumb_bot], show_game = True)
	cont = input("Game Over. Enter c to continue: ")

# Export winner network
biases = [ [list(b) for b in biases_layer] for biases_layer in smart_bot.biases ]
weights = [ [list(w) for w in weights_layer] for weights_layer in smart_bot.weights ]
nNet = {'biases': biases, 'weights': weights}

with open ('[net_params]tictactoe_net.json', 'w') as outfile:
	json.dump(nNet, outfile)