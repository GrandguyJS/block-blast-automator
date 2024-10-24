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

class BlockBlast:
    def __init__(self, log=False, grid=[[0 for _ in range(8)] for _ in range(8)], score=0, combo=1):
        ### Grid
        # [ [Row 1] [Row 2] [Row 3] [Row 4] [Row 5] [Row 6] [Row 7] [Row 8 ]]
        self.grid = grid
        self.score = score
        self.combo = combo
        self.limit = 0
        self.shapes = [7, 15, 31, 1057, 33825, 1082401, 1569, 3105, 225, 57, 7201, 1825, 4231, 1063]

    def __str__(self):
        _str = ""
        for row in self.grid:
            _str += " | ".join(str(cell) for cell in row)
            _str += "\n"
        _str += str(self.score)

        return _str

    def eval(self):
        grid = self.grid
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

        temp = []
        if x <= 3 and y <= 3:
            for i in range(0,5):
                temp.extend(grid[y+i][x:x+5])
        if x > 3 and y <= 3:
            for i in range(0,5):
                temp.extend(grid[y+i][x:])
        if x <= 3 and y > 3:
            for i in range(0,7-y+1):
                temp.extend(grid[y+i][x:x+5])
        if x > 3 and y > 3:
            for i in range(0,7-y+1):
                temp.extend(grid[y+i][x:])

        temp = temp[:len(shape)+1]

        for i in range(0, len(shape)):
            try:
                if shape[i] == 1 and temp[i] == 1:
                    return False
            except:
                return False

        shape.extend([0 for _ in range(25-len(shape))])
        shape = [shape[i:i + 5] for i in range(0, len(shape), 5)]

        if x <= 3 and y <= 3:
            for i in range(0,5):
                grid[y+i][x:x+5] = [shape[i][j] for j in range(5)]
        if x > 3 and y <= 3:
            for i in range(0,5):
                grid[y+i][x:] = [shape[i][j] for j in range(7-x+1)]
        if x <= 3 and y > 3:
            for i in range(0,7-y+1):
                grid[y+i][x:x+5] = [shape[i][j] for j in range(5)]
        if x > 3 and y > 3:
            for i in range(0,7-y+1):
                grid[y+i][x:] = [shape[i][j] for j in range(7-x+1)]

        self.score += sum(bytes_shape)
        
        self.eval()

        return True

    def clear(self):
        self.grid = [[0 for _ in range(8)] for _ in range(8)] if grid is None else grid

    def find_best(self, shape1, shape2, shape3):
        print(shape1, shape2, shape3)
        best = 0
        best_moves = [[None, None, None], [None, None, None], [None, None, None]]

        possibilities = [[shape1, shape2, shape3], [shape1, shape3, shape2], 
                        [shape2, shape1, shape3], [shape2, shape3, shape1], 
                        [shape3, shape1, shape2], [shape3, shape2, shape1]]

        for p in possibilities:
            s1, s2, s3 = p[0], p[1], p[2]
            
            for i1 in range(0, 8):
                for j1 in range(0, 8):
                    _dummy = BlockBlast(grid=copy.deepcopy(self.grid), combo=copy.copy(self.combo), score=copy.copy(self.score))
                    res = _dummy.put(s1, i1, j1)
                    if not res:
                        continue
                    for i2 in range(0, 8):
                        for j2 in range(0, 8):
                            _dummy2 = BlockBlast(grid=copy.deepcopy(_dummy.grid), combo=copy.copy(_dummy.combo), score=copy.copy(_dummy.score))
                            res = _dummy2.put(s2, i2, j2)
                            if not res:
                                continue
                            for i3 in range(0, 8):
                                for j3 in range(0, 8):
                                    _dummy3 = BlockBlast(grid=copy.deepcopy(_dummy2.grid), combo=copy.copy(_dummy2.combo), score=copy.copy(_dummy2.score))
                                    res = _dummy3.put(s3, i3, j3)
                                    if not res:
                                        continue

                                    if best < _dummy3.score:
                                        best = _dummy3.score
                                        best_moves = [[s1, i1, j1], [s2, i2, j2], [s3, i3, j3]]

        for b in best_moves:
            self.put(b[0], b[1], b[2])

    def auto_play(self):
        while True:
            s1, s2, s3 = random.sample(self.shapes, 3)
            self.find_best(s1, s2, s3)
            print(self)