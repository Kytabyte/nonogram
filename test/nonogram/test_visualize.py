import numpy as np

import nonogram as ng


def test_visualize():
    mat = np.array([
        [-1, 0, 1],
        [0, -1, 1],
        [0, 0, 0]
    ])
    ng.show(mat)
