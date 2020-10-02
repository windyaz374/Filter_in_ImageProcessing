import numpy as np 
import math
from dftuv import dftuv
def lpft(type, M,N,D0,n):
    [U,V] = dftuv(M,N)
    D = np.sqrt((U**2+V**2))
    H = np.zeros(D.shape)
    if(type == "ideal"):
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                if(D[i][j]<=D0):
                    H[i][j]=1
    elif(type == "btw"):
        H = 1/(1 + (D/D0)**(2*n))
    elif(type == "gaussian"):
         for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                H[i][j] = math.exp(-(D[i][j]**2)/(2*(D0**2)))
    else:
        print("Unknown filter type")
    return H
