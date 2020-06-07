from .solver import Solver
from .visualize import show


__all__ = ['solve', 'solve_line', 'show']


def solve(rows, cols, mat=None):
    solver_ = Solver(rows, cols, mat)
    return solver_.solve()


solve_line = solver.solve_line
