import numpy as np
from pre_process import return_zeros 
import os

zeros = return_zeros()
#print(zeros)

y_pred = [5,3,7,6,1,9,5,9,8,6,8,6,3,4,8,3,1,7,2,6,6,2,8,4,1,9,5,8,7,9]

digit_paths = os.listdir('images/')
digit_paths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

#print(digit_paths)

def make_grid():
  #grid = [['_' for _ in range(9)] for _ in range(9)]
  grid = np.zeros((9,9))
  grid = grid.astype('str')
  for zero in zeros:
    r = (zero-1) % 9
    c = (zero-1) // 9
    grid[c][r]='_'

  for i in range(len(digit_paths)):
    digit_pos = int(''.join(filter(str.isdigit, digit_paths[i])))
    r = (digit_pos-1) % 9
    c = (digit_pos-1) // 9
    grid[c][r] = y_pred[i]
  return grid

grid=make_grid()

def print_board(bo):
    for i in range(len(bo)):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - - ")

        for j in range(len(bo[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")

            if j == 8:
                print(bo[i][j])
            else:
                print(str(bo[i][j]) + " ", end="")

print_board(grid)