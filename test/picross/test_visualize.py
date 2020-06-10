import numpy as np

import picross as pc


def test_visualize():
    mat = np.array([
        [-1, 0, 1],
        [0, -1, 1],
        [0, 0, 0]
    ])
    pc.show(mat)
