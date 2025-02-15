import numpy as np

# This code is just the python equivalent of the Reversi server functions to flip pieces

def change_colors(row, col, turn, state):
    for incx in range(-1, 2):
        for incy in range(-1, 2):
            if incx == 0 and incy == 0:
                continue
            check_direction(row, col, incx, incy, turn, state)


def check_direction(row, col, incx, incy, turn, state):
    sequence = np.zeros(7, dtype=object)
    seqLen = 0

    for i in range(1, 8):
        r = row + incy * i
        c = col + incx * i
        if r < 0 or r > 7 or c < 0 or c > 7:
            break
        sequence[seqLen] = state[r][c]
        seqLen += 1

    count = 0

    for i in range(0, seqLen):
        if turn == 0:
            if sequence[i] == 2:
                count += 1
            else:
                if sequence[i] == 1 and count > 0:
                    count = 20
                break

        else:
            if sequence[i] == 1:
                count += 1
            else:
                if sequence[i] == 2 and count > 0:
                    count = 20
                break

    if count > 10:
        if turn == 0:
            i = 1
            r = row + incy * i
            c = col + incx * i
            while state[r][c] == 2:
                state[r][c] = 1
                i += 1
                r = row + incy * i
                c = col + incx * i
        else:
            i = 1
            r = row + incy * i
            c = col + incx * i
            while state[r][c] == 1:
                state[r][c] = 2
                i += 1
                r = row + incy * i
                c = col + incx * i
