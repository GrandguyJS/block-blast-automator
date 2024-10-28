# Score
# x + block amount
# 1 row + 10*combo
# 2 row + 10 * combo per row
# 3 row + 10 * combo per row * 2
# 4 row + 10 * combo per row * 3
# 5 row + 10 * combo per row * 4
# lose = 10 * block amount

import random
import copy
from image import draw

class BlockBlast:
    def __init__(self, grid=[[0 for _ in range(8)] for _ in range(8)], score=0, combo=1, log=False):
        ### Grid
        # [ [Row 1] [Row 2] [Row 3] [Row 4] [Row 5] [Row 6] [Row 7] [Row 8 ]]
        self.grid = grid
        self.score = score
        self.combo = combo
        self.limit = 0
        self.log = log

        self.shapes = [1, 3, 33, 7, 1057, 15, 33825, 31, 1082401, 1825, 7201, 4231, 3105, 1569, 225, 57, 2115, 1059, 2145, 195, 51, 561, 4161, 4368, 49, 97, 67, 35]

    def __str__(self):
        _str = ""
        for row in self.grid:
            _str += " | ".join(str(cell) for cell in row)
            _str += "\n"
        _str += str(self.score)

        return _str

    def reset(self):
        self.grid = [[0 for _ in range(8)] for _ in range(8)]
        self.score = 0
        self.combo = 1

    def eval(self):
        grid = self.grid

        if self.log:
            for i in range(0, 8):
                for j in range(0, 8):
                    if grid[i][j] != 0:
                        grid[i][j] = 1


        curr = 0
        # Rows
        for i in range(0, len(grid)):
            if sum(grid[i]) >= 8:
                grid[i] = [0 for _ in range(8)]
                curr += 1

        # Cols
        for i in range(0, len(grid)):
            if sum([row[i] for row in grid]) >= 8:
                for row in grid:
                    row[i] = 0
                curr += 1



        if curr == 1 or curr == 2:
            self.score += 10*self.combo
        elif curr == 3:
            self.score += 10*self.combo*2
        elif curr == 4:
            self.score += 10*self.combo*3
        elif curr == 5:
            self.score += 10*self.combo*4

        if curr == 0:
            self.limit += 1
        else:
            if self.log:
                draw(self.grid)
            self.limit = 0
            self.combo += 1
        
        if self.limit == 3:
            self.combo = 1
            self.limit = 0

        return True

    def put(self, shape, x, y, grid=None):
        grid = self.grid if grid is None else grid

        bytes_shape = [int(bit) for bit in bin(shape)[2:]]
        shape = [0 if bit == 0 else 1 for bit in bytes_shape]
        if self.log:
            for i in range(0, len(shape)):
                shape[i] *= 2

        temp = []
        if x <= 3 and y <= 3:
            for i in range(0,5):
                temp.extend(grid[y+i][x:x+5])
        if x > 3 and y <= 3:
            for i in range(0,5):
                temp.extend(grid[y+i][x:])
                temp.extend([1 for _ in range(x+5-8)])
        if x <= 3 and y > 3:
            for i in range(0,7-y+1):
                temp.extend(grid[y+i][x:x+5])
            for i in range(5-(8-y)):
                temp.extend([1 for _ in range(5)])
        if x > 3 and y > 3:
            for i in range(0,7-y+1):
                temp.extend(grid[y+i][x:])
                temp.extend([1 for _ in range(x+5-8)])
            for i in range(5-(8-y)):
                temp.extend([1 for _ in range(5)])

        temp = temp[:len(shape)+1]

        for i in range(0, len(shape)):
            try:
                if shape[i] != 0 and temp[i] != 0:
                    return False
            except:
                return False

        shape.extend([0 for _ in range(25-len(shape))])
        shape = [shape[i:i + 5] for i in range(0, len(shape), 5)]

        if x <= 3 and y <= 3:
            for i in range(0,5):
                for j in range(0,5):
                    if shape[i][j] > 0:
                        grid[y+i][x+j] = shape[i][j]
        if x > 3 and y <= 3:
            for i in range(0,5):
                for j in range(0,7-x):
                    if shape[i][j] > 0:
                        grid[y+i][x+j] = shape[i][j]
        if x <= 3 and y > 3:
            for i in range(0,7-y+1):
                for j in range(0,5):
                    if shape[i][j] > 0:
                        grid[y+i][x+j] = shape[i][j]
        if x > 3 and y > 3:
            for i in range(0,7-y+1):
                for j in range(0,7-x):
                    if shape[i][j] > 0:
                        grid[y+i][x+j] = shape[i][j]

        self.score += sum(bytes_shape)
        
        if self.log:
            draw(self.grid)

        self.eval()

        return True

    def clear(self):
        self.grid = [[0 for _ in range(8)] for _ in range(8)] if grid is None else grid

    def find_best(self, shape1, shape2, shape3):
        best = 0
        best_moves = [[None, None, None], [None, None, None], [None, None, None]]

        possibilities = [[shape1, shape2, shape3], [shape1, shape3, shape2], 
                        [shape2, shape1, shape3], [shape2, shape3, shape1], 
                        [shape3, shape1, shape2], [shape3, shape2, shape1]]

        for p in possibilities:
            s1, s2, s3 = p[0], p[1], p[2]
            
            for i1 in range(0, 8):
                for j1 in range(0, 8):
                    _dummy = BlockBlast(grid=copy.deepcopy(self.grid), combo=copy.copy(self.combo), score=copy.copy(self.score), log=False)
                    res = _dummy.put(s1, i1, j1)
                    if not res:
                        continue
                    for i2 in range(0, 8):
                        for j2 in range(0, 8):
                            _dummy2 = BlockBlast(grid=copy.deepcopy(_dummy.grid), combo=copy.copy(_dummy.combo), score=copy.copy(_dummy.score), log=False)
                            res = _dummy2.put(s2, i2, j2)
                            if not res:
                                continue
                            for i3 in range(0, 8):
                                for j3 in range(0, 8):
                                    _dummy3 = BlockBlast(grid=copy.deepcopy(_dummy2.grid), combo=copy.copy(_dummy2.combo), score=copy.copy(_dummy2.score), log=False)
                                    res = _dummy3.put(s3, i3, j3)
                                    if not res:
                                        continue

                                    if best < _dummy3.score:
                                        best = _dummy3.score
                                        best_moves = [[s1, i1, j1], [s2, i2, j2], [s3, i3, j3]]

        if best_moves[0][0] == None:
            print("Lost the game!")
            return False

        for b in best_moves:
            self.put(b[0], b[1], b[2])
        return True

    def auto_play(self):
        while self.find_best(random.sample(self.shapes, 1)[0], random.sample(self.shapes, 1)[0], random.sample(self.shapes, 1)[0]):
            print(self)
        return