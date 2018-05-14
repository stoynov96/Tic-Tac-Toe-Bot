from numpy import argmax
from numpy import reshape

# Convets board data ot a format usable as an input layer of the neural network
# <param> board:	1d list containing all board marks
def convert_board(board, my_player_id, other_player_id):
	n = len(board)
	# first n layers are my (the bot's) marks
	first = [int(cell == my_player_id) for cell in board]

	# next n layers are their (the player's) marks
	last = [int(cell == other_player_id) for cell in board]

	return reshape( first+last , (len(first)+len(last), 1) )

# Extracts a cell id to play from a neural network output
def extract_cell_id(activation):
	return argmax(activation)