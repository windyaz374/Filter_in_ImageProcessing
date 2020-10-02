import numpy as np
from notch import notch
import cv2
from paddedsize import paddedsize
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift, ifft2

football = cv2.imread("halftone.png", 0)
plt.imshow(football, cmap = 'gray')
H, W = np.shape(football)
hW = W//2
hH = H//2
PQ = paddedsize(np.array(np.shape(football)))
x = []
y = []
for i in range(133, PQ[0], 266):
   for j in range(200, PQ[1], 400):
     if not (i == 665 and j == 1000):
        y.append(i)
        x.append(j)
for i in range(0, PQ[0], 266):
   for j in range(0, PQ[1], 400):
     if not (i == 665 and j == 1000):
        y.append(i)
        x.append(j)

notch_ft = []
for i in range(len(x)):
    h = notch('gaussian', PQ[0], PQ[1], 5, x[i], y[i], 1)
    notch_ft.append(h)

F = fft2((football)/(W*H), s= [2*H, 2*W])
F_ifft = F
for i in range(len(notch_ft)):
    F_ifft = F_ifft*notch_ft[i]
F_football = ifft2(F_ifft)
F_football = F_football[:H,:W]

S1 = np.log(1+ np.abs(F))
S2 = np.log(1+ np.abs(F_ifft))
images = [football, np.abs(F_football), S1, S2]
title = ['Anh co nhieu', 'Anh da loc nhieu', 'Pho anh co nhieu', 'Pho anh loc nhieu']
plt.imshow(S1, cmap='gray')
plt.show()
plt.imshow(S2, cmap='gray')
plt.show()
for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], cmap='gray')
    plt.title(title[i])
plt.show()

