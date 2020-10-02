import numpy as np
from notch import notch
import cv2
from paddedsize import paddedsize
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift, ifft2

football = cv2.imread("noiseball.png", 0)
H, W = np.shape(football)
hW = W//2
hH = H//2
PQ = paddedsize(np.array(np.shape(football)))

H1 = notch('gaussian', PQ[0], PQ[1], 10, 50, 100, 1)
H2 = notch('gaussian', PQ[0], PQ[1], 10, 0, 400, 1)
H3 = notch('gaussian', PQ[0], PQ[1], 10, 620, 100, 1)
H4 = notch('gaussian', PQ[0], PQ[1], 10, 22, 414, 1)
H5 = notch('gaussian', PQ[0], PQ[1], 10, 592, 414, 1)
H6 = notch('gaussian', PQ[0], PQ[1], 10, 0 , 114, 1)

F = fft2((football), s= [2*H, 2*W])/(H*W)
F_ifft = F*H1*H2*H3*H4*H5*H6
F_football = ifft2(F_ifft)
F_football = F_football[:H,:W]

F = fftshift(F)
F_ifft = fftshift(F_ifft)
S1 = np.log(1+ np.abs(F))
S2 = np.log(1+ np.abs(F_ifft))
images = [football, np.abs(F_football), S1, S2]
title = ['Anh co nhieu', 'Anh da loc nhieu', 'Pho anh co nhieu', 'Pho anh loc nhieu']
r = 150
for i in range(4):
    if i < 2:
        plt.subplot(2, 2, i+1), plt.imshow(images[i], cmap = 'gray')
    else:
        plt.subplot(2, 2, i+1), plt.imshow(images[i][hH*2-r:hH*2+r,hW*2-r:hW*2+r], extent=[-r,r,-r,r], cmap='gray')
    plt.title(title[i])
plt.show()