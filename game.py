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
	cell_id = datap.extract_cell_id(cell_id) # extract the cell id from the output

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

	# Execution Loop
	while True:

		# Make sure no more than two players are playing
		try:
			if len(bots) != 2:
				raise ValueError("Exactly two bots must play tic tac toe")
		except ValueError as va:
			raise

		# Bots play their moves
		for i in range(len(bots)):
			success, won = bot_move(bots[i], gs.BOT_IDS[i], gs.BOT_IDS[1-i])

			if show_game: 
				board.display()
				input()

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
