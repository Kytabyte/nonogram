# picross

A fast [nonogram/picross](https://en.wikipedia.org/wiki/Nonogram) solver written in Python. Nonogram/Picross is a logic puzzles to meet 
requirements by filling some grids in a rectangular gridded board.

# Usage

Give the given `rows` and `cols` as 2d array. Solve the problem and show the returned board

```python

import picross as pc

rows = [[4], [3, 3], [2, 1], [2, 2], [1, 2, 3], [2, 3], [4, 1], [10], [8], [4]]
cols = [[6], [3, 4], [1, 1, 3], [2, 1, 4], [1, 1, 3], [1, 1, 3], [2, 2, 3], [1, 1, 2], [1, 2, 2], [2, 2]]

board = pc.solve(rows, cols)
pc.show(board)

```
