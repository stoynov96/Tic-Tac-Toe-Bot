
# =================
# === Cnostants ===
# =================


class GameSettings:
    VOID_CELL = 0

    player_count = 2

class LineScore:
    # Initializes a line score (this can be column, row or diagonal score)
    # <param> player_id:    ID of the player that has played on this line.
    #                       If another player plays on this line, no one can win there
    def __init__(self, init_score, player_id):
        self.score = init_score
        self.player_id = player_id

    # Resets the line score (this can be column, row or diagonal score)
    # <param> player_id:    ID of the player that has played on this line.
    #                       If another player plays on this line, no one can win there
    def reset_score(self, init_score, player_id):
        self.score = init_score
        self.player_id = player_id

    # Adds score to a specific line
    def add_score(self, player_id, score_amount = 1):
        self.score += score_amount
        # If a new player has played in this line, reset so no win can be obtained
        if self.player_id == GameSettings.VOID_CELL:
            self.player_id = player_id
        elif player_id != self.player_id:
            self.score = -1


class Board(object):
    
    # Initializes the board with a specific size of one dimension. The board is always a square.
    def __init__(self, size):
        self.size = size
        self.board = [GameSettings.VOID_CELL for i in range(size**2)]
        # Number of non-void cells
        self.filled_count = 0
        # Column (score , dominating player id a.k.a player that uniquely has cells in that column)
        self.col_score = [LineScore(0, GameSettings.VOID_CELL) for i in range(size)]
        # Row score (score , dominating player id)
        self.row_score = [LineScore(0,GameSettings.VOID_CELL) for i in range(size)]
        # Diagonal Score (score , dominating player id)
        # size = 2. Main and Secondary. Others are not counted in Tic Tac Toe
        self.diag_score = [LineScore(0,GameSettings.VOID_CELL), LineScore(0,GameSettings.VOID_CELL)]

    # Clears board to prepare for a new game
    # Using this saves memory allocation overhead
    def clear(self):
        # Reset number of non-void cells
        self.filled_count = 0
        # Clear board marks
        for i in range(len(self.board)):
            self.board[i] = GameSettings.VOID_CELL
        # Clear row and column score
        for i in range(self.size):
            self.col_score[i].reset_score(0, GameSettings.VOID_CELL)
            self.row_score[i].reset_score(0, GameSettings.VOID_CELL)
        # Clear diagonal scores
        self.diag_score[0].reset_score(0, GameSettings.VOID_CELL)
        self.diag_score[1].reset_score(0, GameSettings.VOID_CELL)




    # Fills a specific cell with a specified player id's mark
    # <return> if the cell has been filled out successfully
    def fill(self, cell_id, player_id):
        try:    
            if player_id > GameSettings.player_count:
                raise ValueError('Cannot fill cell. Player Id invalid. Not enough players.')
        
        except ValueError as va:
            print(va.args)

        except:
            print("Exception: Player Id not an integer")
            return False, False

        # Fail to fill if cell is not void
        if self.board[cell_id] != GameSettings.VOID_CELL:
            return False, False

        # Obtain row and column from cell id
        row = cell_id // self.size
        col = cell_id % self.size

        # Add score for all lines affected
        self.add_score(row, col, player_id)

        # Fill cell and return with success
        self.board[cell_id] = player_id

        # Update non-void cell count
        self.filled_count += 1

        # Return successfully, check if player won
        return True, self.check_win(row, col)

    def add_score(self, row, col, player_id):
        # == Add ==
        # Column, Row
        self.col_score[col].add_score(player_id)
        self.row_score[row].add_score(player_id)

        # Main diagonal
        if row == col: self.diag_score[0].add_score(player_id)
        # Secondary diagonal
        if self.size - row - 1 == col: self.diag_score[1].add_score(player_id)

    # Checks if enough points have been accumulated around a specific cell
    def check_win(self, row, col):
        rowcol = self.col_score[col].score >= self.size or self.row_score[row].score >= self.size
        # It's faster to just check diagonals every time than to determine if they should be checked
        diags = self.diag_score[0].score >= self.size or self.diag_score[1].score >= self.size
        return rowcol or diags

    # Checks if the board has no void cells
    def is_full(self):
        return self.filled_count >= len(self.board)


    # Displays the current state of the board complete with players markings on it
    def display(self, indentation = ''):
        for i in range(self.size):
            print(indentation + str(self.board[i*self.size : (i+1)*self.size]))

    # Dumps debug information
    def DEBUG_dump(self):
        print('Column Score: {0}\nRow Score:    {1}\nDiag Score:   {2}'.format(
            [(c.score, c.player_id) for c in self.col_score], 
            [(r.score, r.player_id) for r in self.row_score], 
            [(d.score, d.player_id) for d in self.diag_score])
        )