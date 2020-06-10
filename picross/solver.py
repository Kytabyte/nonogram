"""

"""

import collections
import functools
import random

import numpy as np

FILL, EMPTY, UNDEF = 1, 0, -1


SolLnRetType = collections.namedtuple('SolLnRetType', ['valid', 'change', 'full', 'pat_range'])
_SolLnDPRetType = collections.namedtuple('_SolLnDPRetType', ['valid', 'filled', 'empty'])


# O(N^2 * K) time
# O(N^2 * K) space
def solve_line(line, pat):
    """ solve a single line `line` with pattern `pat`, will mark all determined filled and empty sell

    :param line: a list of state
    :param pat: the constraints the `line` need to satisfy
    :return: (is_solvable, changed_index, is_line_full, each pattern's range)
    """
    n, m = len(line), len(pat)

    # pat[j]'s range of placement
    pat_range = [[n - 1, 0] for _ in range(m)]

    @functools.lru_cache(None)
    def dp(i, j):
        if i >= n and j >= m:
            return _SolLnDPRetType(True, set(), set())

        if i >= n:
            return _SolLnDPRetType(False, set(), set())

        if j >= m:
            if np.any(line[i:] == FILL):
                return _SolLnDPRetType(False, set(), set())
            return _SolLnDPRetType(True, set(), set(range(i, n)))

        # naming convention:
        # A_B where A in {'fill', 'empty'}, B in _SolLnDPRetType.names
        # A means the state of this cell (make this cell filled or empty)
        # B means the properties of state A: if state A is valid,
        #   what are the must-filled cells and must-empty cells under state A

        # is line[i:i+pat[j]] valid to be all filled?
        fill_valid, fill_filled, fill_empty = False, set(), set()
        if (i+pat[j] == n or i+pat[j] < n and line[i+pat[j]] != FILL) and np.all(line[i:i+pat[j]] != EMPTY):
            fill_dp = dp(i + pat[j] + 1, j + 1)
            if fill_dp.valid:
                fill_valid = True
                fill_filled = fill_dp.filled | set(range(i, i + pat[j]))
                if i + pat[j] < n:
                    fill_empty = fill_dp.empty | {i + pat[j]}

                # update the pat[j]'s range of placement
                pat_range[j][0] = min(pat_range[j][0], i)
                pat_range[j][1] = max(pat_range[j][1], i)

        # is line[i] valid to be empty?
        empty_valid, empty_filled, empty_empty = False, set(), set()
        if line[i] != FILL:
            empty_dp = dp(i + 1, j)
            if empty_dp.valid:
                empty_valid = True
                empty_filled |= empty_dp.filled
                empty_empty = empty_dp.empty | {i}

        # integrate both fill and empty cases to find the deterministic cells
        if fill_valid and empty_valid:
            return _SolLnDPRetType(
                True,
                fill_filled & empty_filled,
                fill_empty & empty_empty
            )

        return _SolLnDPRetType(
            fill_valid or empty_valid,
            fill_filled or empty_filled,
            fill_empty or empty_empty
        )

    # solve line with dp
    solve_ret = dp(0, 0)

    if not solve_ret.valid:
        return SolLnRetType(False, set(), False, pat_range)

    change = set()
    for i in solve_ret.filled:
        if line[i] == EMPTY:
            raise RuntimeError('Internal Error, please contact author')
        if line[i] != FILL:
            line[i] = FILL
            change.add(i)

    for i in solve_ret.empty:
        if line[i] == FILL:
            raise RuntimeError('Internal Error, please contact author')
        if line[i] != EMPTY:
            line[i] = EMPTY
            change.add(i)

    return SolLnRetType(True, change, all(line[i] >= 0 for i in range(n)), pat_range)


_SolInfo = collections.namedtuple('_SolveInputType', ('to_solve', 'remain', 'pat_range'))


class Solver:
    def __init__(self, rows, cols, mat=None):
        self._nrow, self._ncol = len(rows), len(cols)
        self._row_pats, self._col_pats = rows, cols
        self._mat = np.ones((self._nrow, self._ncol)) * -1 if mat is None else np.array(mat)

        if self._mat.shape != (self._nrow, self._ncol):
            raise RuntimeError('Matrix shape does not match the size of "rows" and "cols".')

        # stats
        self._cnt = self._depth = 0

        # other
        self._verbose = False

    def verbose(self, state):
        """ whether print some debug info

        :param state: bool, print or not
        :return: None
        """
        self._verbose = state

    def solve(self):
        """

        :return:
        """
        nrow, ncol = self._nrow, self._ncol

        rows = _SolInfo(set(range(nrow)), set(range(nrow)), [[] for _ in range(nrow)])
        cols = _SolInfo(set(range(ncol)), set(range(ncol)), [[] for _ in range(ncol)])
        self._solve(rows, cols)

        return self._mat

    def _solve(self, rows, cols):
        self._cnt += 1
        if self._verbose and self._cnt % 200 == 0:
            print(self._cnt, end=' ')

        mat = self._mat
        row_pats, col_pats = self._row_pats, self._col_pats

        updated = set()
        swapped = False

        def rollback():
            for r, c in updated:
                mat[r][c] = UNDEF

        # do bfs on rows and cols alternately
        while rows.to_solve:
            for r in rows.to_solve:
                solve_ln_ret = solve_line(mat[r], row_pats[r])

                # line cannot satisfy the given pattern
                if not solve_ln_ret.valid:
                    if swapped:
                        mat = mat.T
                    rollback()
                    return False

                for c in solve_ln_ret.change:
                    cols.to_solve.add(c)
                    updated.add((r, c) if not swapped else (c, r))
                if solve_ln_ret.full:
                    rows.remain.remove(r)
                rows.pat_range[r] = solve_ln_ret.pat_range

            rows.to_solve.clear()

            rows, cols = cols, rows
            swapped = not swapped
            row_pats, col_pats = col_pats, row_pats
            mat = mat.T

        # no cells can be marked for sure. Check if all rows and cols are solved
        if not rows.remain and not cols.remain:
            return True

        if swapped:
            rows, cols = cols, rows
            mat = mat.T
            swapped = not swapped
            row_pats, col_pats = col_pats, row_pats

        # problem is not solved yet. Try some assumptions and do backtrack
        for filled, empty in self._gen_backtrack(rows.pat_range, cols.pat_range):
            nxt_row = set(filled[0]) | set(empty[0])
            nxt_col = set(filled[1]) | set(empty[1])

            mat[filled], mat[empty] = FILL, EMPTY
            if self._solve(_SolInfo(nxt_row, set(rows.remain), list(rows.pat_range)),
                           _SolInfo(nxt_col, set(cols.remain), list(cols.pat_range))):
                return True
            mat[filled], mat[empty] = UNDEF, UNDEF

        rollback()
        return False

    def _gen_backtrack(self, row_pats_range, col_pats_range):
        # generate a series of backtrack cases
        # yield a list of indices to be ones and a list of indices to be zeros
        # (ones_row, ones_col), (zeros_row, zeros_col)

        # current algorithm is to find the pattern with fewest possibilities
        # pats_range stores the range of each pattern can be placed.
        # This choice is good for hard problem to be solved in a acceptable time,
        # but not very efficient for well-determined cases.

        mat = self._mat
        nrow, ncol = self._nrow, self._ncol
        row_pats, col_pats = self._row_pats, self._col_pats

        # row with min possible ranges
        row_min = {
            'ln_idx': -1,
            'pat_idx': -1,
            'len': nrow,
            'range': (0, nrow-1),
        }
        for r in range(nrow):
            pat_ranges = row_pats_range[r]
            for i, (left, right) in enumerate(pat_ranges):
                if 0 < right - left < row_min['len']:
                    row_min['ln_idx'] = r
                    row_min['pat_idx'] = i
                    row_min['len'] = right - left + 1
                    row_min['range'] = (left, right)

        # col with min possible ranges
        col_min = {
            'ln_idx': -1,
            'pat_idx': -1,
            'len': ncol,
            'range': (0, ncol - 1),
        }
        for c in range(ncol):
            pat_ranges = col_pats_range[c]
            for i, (left, right) in enumerate(pat_ranges):
                if 0 < right - left < col_min['len']:
                    col_min['ln_idx'] = c
                    col_min['pat_idx'] = i
                    col_min['len'] = right - left + 1
                    col_min['range'] = (left, right)

        # compare row min and col min
        if row_min['len'] < col_min['len']:
            line = mat[row_min['ln_idx']]
            pat = row_pats[row_min['ln_idx']][row_min['pat_idx']]
            pat_range = row_min['range']
        else:
            line = mat[:, col_min['ln_idx']]
            pat = col_pats[col_min['ln_idx']][col_min['pat_idx']]
            pat_range = col_min['range']

        # start indices of all possibilities
        starts = list(range(pat_range[0], pat_range[1]+1))
        random.shuffle(starts)  # add some randomness
        line_len = len(line)
        for s in starts:
            e = s + pat

            if (np.any(line[s:e]) == EMPTY) or (s > 0 and line[s-1] == FILL) or (e < line_len and line[e] == FILL):
                continue

            filled = [i for i in range(s, e) if line[i] == UNDEF]
            empty = [i for i in (s-1, e) if 0 <= i < line_len and line[i] == UNDEF]

            if not filled and not empty:
                continue

            if row_min['len'] < col_min['len']:
                yield ([row_min['ln_idx']], filled), ([row_min['ln_idx']], empty)
            else:
                yield (filled, [col_min['ln_idx']]), (empty, [col_min['ln_idx']])
