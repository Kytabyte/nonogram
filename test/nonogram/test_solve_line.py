import numpy as np
import pytest

import nonogram as ng


def _do_check(line, pat, expected, solvable=True):
    line, expected = np.array(line), np.array(expected)
    ret = ng.solve_line(line, pat)
    assert ret.valid == solvable
    assert np.all(line == expected)


# some simple misc cases
def test_simple_group_1():
    line = [-1, -1, -1, 1, -1]
    pat = [2]
    expected = [0, 0, -1, 1, -1]
    _do_check(line, pat, expected)


# fill marks
def test_fill_mark_1():
    pass


# fill empty
def test_fill_empty_1():
    line = [-1] * 18 + [1] * 5 + [-1] * 3 + [1] * 5 + [-1] * 9
    pattern = [3, 1, 6, 7]
    expected = [-1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, 1, 1, 1, 1,
                1, -1, -1, -1, 1, 1, 1, 1,
                1, -1, -1, 0, 0, 0, 0, 0, 0,
                0]
    _do_check(line, pattern, expected)


# robustness test. Program should return the deterministic results
# for all partially solved cases
def test_robustness_1():
    line1 = [-1] * 11 + [0] * 1 + [1] * 11 + [0] * 3 + [1] * 6 + [-1] * 8
    line2 = [-1] * 11 + [0] * 1 + [1] * 11 + [-1] * 3 + [1] * 6 + [-1] * 8
    line3 = [-1] * 11 + [-1] * 1 + [1] * 11 + [-1] * 3 + [1] * 6 + [-1] * 8
    pattern = [4, 4, 11, 6]
    expected = [-1, -1, 1, 1, -1, -1,
                -1, 1, 1, -1, -1, 0,
                1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 0,
                0, 0, 0, 0, 0, 0, 0]
    _do_check(line1, pattern, expected)
    _do_check(line2, pattern, expected)
    _do_check(line3, pattern, expected)


# union of marked cell
def test_union_1():
    line = [-1] * 18 + [0, 0] + [1] * 8 + [-1] + [1, 1] + [-1] * 9
    pattern = [2, 7, 11]
    expected = [-1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, 0, 0, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0]
    _do_check(line, pattern, expected)


def test_union_2():
    line = [-1] * 14 + [0] + [1] + [-1, -1, 1, -1, 1, 1] + [-1] * 17
    pattern = [7, 3, 4, 1]
    expected = [0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 1, 1, 1,
                1, 1, 1, 1, 0, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1]
    _do_check(line, pattern, expected)


# unsolvable cases
def test_unsolvable_1():
    line = [1] + [-1] * 2 + [1] + [-1] * 5 + [1] + [-1] * 5 + [1] + [-1] * 4
    pattern = [3, 3, 3]
    expected = [1] + [-1] * 2 + [1] + [-1] * 5 + [1] + [-1] * 5 + [1] + [-1] * 4
    _do_check(line, pattern, expected, False)
