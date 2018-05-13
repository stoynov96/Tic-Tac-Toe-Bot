import tictactoe as ttt
import os
import random
import network
import data_processing as datap

clear = lambda: os.system('cls')

# Game Settings
class gs:
	PLAYER_ID = 1
	BOT_ID = 2


# Initialize game board
board = ttt.Board(int(input("Enter size of the board: ")))

# Initialize bot neural network
bot_net = network.Network([len(board.board)*2,30,10,9])

# Prompt
def prompt():

	print("Board cells numbers:")
	[print([i*board.size + j + 1 for j in range(board.size)]) for i in range(board.size)]

	print("Current board state:")
	board.display()


# <return> if player won
def player_move():

	success, won = False, False

	while not success:
		clear()
		prompt()
		cell_id = int(input()) - 1
		try:
			success, won = board.fill(cell_id, gs.PLAYER_ID)
		except IndexError as ie:
			print( "Could not fill cell. Please enter a valid cell id between {0} and {1}".format(0, len(board.board)) )

	return won


# <return> if bot won
def bot_move():
	cell_id = bot_net.feedforward(datap.convert_board(board.board, gs.BOT_ID, gs.PLAYER_ID))
	cell_id = datap.extract_cell_id(cell_id) # extract the cell id from the output
	input('Bot played cell #{0}\n'.format(cell_id+1))
	success, won = board.fill(cell_id, gs.BOT_ID)
	if not success:
		print("Player won")
		exit()

	return won


# Execution Loop
clear()
while True:
	# Player plays a move
	won = player_move()
	if won:
		board.display()
		print('Player won')
		break

	# Bot plays a move
	won = bot_move()
	if won:
		board.display()
		print('Bot won')
		break

	# clear()
	# board.DEBUG_dump()