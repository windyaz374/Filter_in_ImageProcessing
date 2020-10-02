import numpy as np
from matplotlib import pyplot as plt
from lpft import lpft
LPF_ideal=np.fft.fftshift(lpft('ideal', 500, 500, 50,1))
plt.title('Ideal Lowpass Filter'), plt.imshow(LPF_ideal, cmap = 'gray')
plt.show()
LPF_ideal=np.fft.fftshift(lpft('btw', 500, 500, 50,1))
plt.title('Btw Lowpass Filter'), plt.imshow(LPF_ideal, cmap = 'gray')
plt.show()
LPF_ideal=np.fft.fftshift(lpft('gaussian', 500, 500, 50,1))
plt.title('Gaussian Lowpass Filter'), plt.imshow(LPF_ideal, cmap = 'gray')
plt.show()

