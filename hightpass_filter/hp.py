import numpy as np
from matplotlib import pyplot as plt
from hpft import hpft
HPF_ideal=np.fft.fftshift(hpft('ideal', 500, 500, 50,1))
plt.title('Ideal Highpass Filter'), plt.imshow(HPF_ideal, cmap = 'gray')
plt.show()
HPF_ideal=np.fft.fftshift(hpft('btw', 500, 500, 50,1))
plt.title('Ideal Highpass Filter'), plt.imshow(HPF_ideal, cmap = 'gray')
plt.show()
HPF_ideal=np.fft.fftshift(hpft('gaussian', 500, 500, 50,1))
plt.title('Gaussian Highpass Filter'), plt.imshow(HPF_ideal, cmap = 'gray')
plt.show()