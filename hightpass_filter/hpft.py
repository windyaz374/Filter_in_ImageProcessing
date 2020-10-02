import numpy as np
from lpft import lpft
def hpft(type, M,N,D0,n):
    Hlp = lpft(type, M, N, D0, n)
    Hp = 1 - Hlp
    return Hp
