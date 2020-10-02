import numpy as np
def dftuv(M,N):
    u = np.linspace(0,M-1,M)
    v = np.linspace(0,N-1,N)
    for i in np.int16(u):
        if (i>M/2):
            u[i]=u[i]-M
    for i in np.int16(v):
        if (i>N/2):
            v[i]=v[i]-N
    [V,U] = np.meshgrid(v,u)
    return [V,U]
