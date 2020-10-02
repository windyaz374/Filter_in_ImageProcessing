import numpy as np
from lpft import lpft
def notch(type, M, N, D0, x, y, n):
    Hlp = lpft(type, M, N, D0, n)
    H = 1 - Hlp
    H = np.roll(H, y-1, axis =0)
    H = np.roll(H, x-1, axis =1)
    return H