import tictactoe as ttt
import os
import random

clear = lambda: os.system('cls')

# Game Settings
class gs:
	PLAYER_ID = 1
	BOT_ID = 2


# Initialize game board
board = ttt.Board(int(input("Enter size of the board: ")))

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
	cell_id = random.randint(0, len(board.board)-1)
	success, won = False, False
	while not success:
		success, won = board.fill(cell_id, gs.BOT_ID)
		cell_id = random.randint(0, len(board.board)-1)

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