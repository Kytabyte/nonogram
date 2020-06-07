""" show matrix """


_CODE = {-1: '\u2b1c', 0: '\u274c', 1: '\u2b1b'}


wl = lambda x: ''.join(map(_CODE.__getitem__, x))


def show(mat):
    for i, row in enumerate(mat):
        print(wl(row))
