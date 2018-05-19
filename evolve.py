import evolution
import game
import json



GENE_POOL_COUNT = 1
GENE_POOL_SIZE = 100

bot_gene_pool = evolution.BotEvolution(GENE_POOL_SIZE, GENE_POOL_COUNT)


cont = 'c'
while (cont == 'c'):
	fittest = bot_gene_pool.evolve(update_generation = 20)

	try:
		smart_bot = fittest[0].genome
		dumb_bot = fittest[ max(0, len(fittest)-1) ].genome
	except IndexError as ie:
		print("Invalid gene pool index. Please choose a valid gene pool")
		continue


	print('Bots game:')
	game.play_game([smart_bot, dumb_bot], show_game = True)
	cont = input("Game Over. Enter c to continue: ")

# Export winner network
biases = [ [list(b) for b in biases_layer] for biases_layer in smart_bot.biases ]
weights = [ [list(w) for w in weights_layer] for weights_layer in smart_bot.weights ]
nNet = {'biases': biases, 'weights': weights}

with open ('[net_params]tictactoe_net.json', 'w') as outfile:
	json.dump(nNet, outfile)