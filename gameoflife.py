import numpy as np
import time
import os

def display(grid):
    os.system('clear')
    for row in grid:
        for cell in row:
            print("■" if cell else " ", end="")
        print()
    print()

def game_of_life(grid, generations):
    for _ in range(generations):
        new_grid = np.copy(grid)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                total = int((grid[i, (j-1)%grid.shape[1]] + grid[i, (j+1)%grid.shape[1]] +
                             grid[(i-1)%grid.shape[0], j] + grid[(i+1)%grid.shape[0], j] +
                             grid[(i-1)%grid.shape[0], (j-1)%grid.shape[1]] + grid[(i-1)%grid.shape[0], (j+1)%grid.shape[1]] +
                             grid[(i+1)%grid.shape[0], (j-1)%grid.shape[1]] + grid[(i+1)%grid.shape[0], (j+1)%grid.shape[1]]))

                if grid[i, j] == 1:
                    if (total < 2) or (total > 3):
                        new_grid[i, j] = 0
                else:
                    if total == 3:
                        new_grid[i, j] = 1
        grid = new_grid
        display(grid)
        time.sleep(0.3)

if __name__ == "__main__":
    rows, cols = 20, 40
    grid = np.random.randint(2, size=(rows, cols))
    generations = 100

    game_of_life(grid, generations)