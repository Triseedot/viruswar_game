# Parameters
height = 10
width = 8
movesCount = 3


class Cell:
    def __init__(self, is_free=True, is_alive=False, player=None):
        self.isFree = is_free
        self.isAlive = is_alive
        self.player = player


class Instance:
    def __init__(self):
        self.field = [[Cell() for _ in range(width)] for _ in range(height)]  # Game board
        self.isActive = [[False for _ in range(width)] for _ in range(height)]

        self.currentPlayer = 0
        self.movesLeft = movesCount

        self.field[0][0] = Cell(is_free=False, is_alive=True, player=0)
        self.field[height - 1][width - 1] = Cell(is_free=False, is_alive=True, player=1)

        self.calc_active()

    def get(self, x, y):
        if self.field[x][y].isFree:
            return " "
        elif self.field[x][y].player == 0:
            if self.field[x][y].isAlive:
                return "🔵"
            else:
                return "🟦"
        else:
            if self.field[x][y].isAlive:
                return "🔴"
            else:
                return "🟥"

    def make_active(self, x, y):
        if x < 0 or x >= height or y < 0 or y >= width or self.isActive[x][y]:
            return
        self.isActive[x][y] = True
        if self.field[x][y].isFree or self.field[x][y].player != self.currentPlayer:
            return

        self.make_active(x + 1, y)
        self.make_active(x - 1, y)
        self.make_active(x, y + 1)
        self.make_active(x, y - 1)

    def calc_active(self):
        for x in range(height):
            for y in range(width):
                self.isActive[x][y] = False

        for x in range(height):
            for y in range(width):
                cell = self.field[x][y]
                if not cell.isFree and cell.player == self.currentPlayer and cell.isAlive:
                    self.make_active(x, y)

    def move(self, move_x, move_y):
        x = move_x
        y = move_y

        if (not self.isActive[x][y] or not self.field[x][y].isFree and
                (self.field[x][y].player == self.currentPlayer or not self.field[x][y].isAlive)):
            return False

        if self.field[x][y].isFree:
            self.field[x][y] = Cell(is_free=False, is_alive=True, player=self.currentPlayer)
        else:
            self.field[x][y] = Cell(is_free=False, is_alive=False, player=self.currentPlayer)

        self.movesLeft -= 1
        if self.movesLeft == 0:
            self.currentPlayer = (self.currentPlayer + 1) % 2
            self.movesLeft = movesCount

        self.calc_active()

        return True

    def is_over(self):
        for x in range(height):
            for y in range(width):
                if not self.isActive[x][y]:
                    continue
                cell = self.field[x][y]
                if cell.isFree or (cell.player != self.currentPlayer and cell.isAlive):
                    return False
        return True
