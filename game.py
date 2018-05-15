import tictactoe as ttt
import os
import data_processing as datap

clear = lambda: os.system('cls')

# Game Settings
class gs:

	BOARD_SIZE = 3

	BOT_COUNT = 2 # 1 if player. 2 if no player
	BOT_IDS = [1,2]


# <return> if bot won
def bot_move(bot_net, player_id, opponent_id):
	cell_id = bot_net.feedforward(datap.convert_board(board.board, player_id, opponent_id))
	cell_id = datap.extract_cell_id(cell_id) # extract the cell id from the neural network's output

	success, won = board.fill(cell_id, player_id)

	return success, won



# Initialize game board
# board = ttt.Board(int(input("Enter size of the board: ")))
board = ttt.Board(gs.BOARD_SIZE)


# <param> bots:	list of length 2 - the two bots that will play against each other
# <return> id of the bot that won the game. negative ID if some bot was disqualified for breaking the rules
def play_game(bots, show_game = False):

	# Clear board for game
	board.clear()

	# Make sure no more than two players are playing
	try:
		if len(bots) != 2:
			raise ValueError("{0} bots cannot play. Exactly two bots must play tic tac toe".format(len(bots)))
	except ValueError as va:
		raise

	# Execution Loop
	while True:

		# Bots play their moves
		for i in range(len(bots)):
			success, won = bot_move(bots[i], gs.BOT_IDS[i], gs.BOT_IDS[1-i])

			if show_game: 
				board.display()
				input()

			if is_standstill(board): # Draw
				return 0

			if won:
				if show_game:
					board.display()
					print('Bot {0} won'.format(i))
				return gs.BOT_IDS[i]
			if not success:
				if show_game:
					board.display()
					print('Bot {0} lost'.format(i))
				return -gs.BOT_IDS[i]

# Checks if a board is at a standstill
def is_standstill(board):
	# If the board is full, it is at a standstill
	return board.is_full()
