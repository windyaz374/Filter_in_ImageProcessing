import numpy as np
from matplotlib import pyplot as plt
from notch import notch
Notch_ft=np.fft.fftshift(notch('ideal', 500, 500, 50,50,100,1))
plt.title('Ideal Notch Filter'), plt.imshow(Notch_ft, cmap = 'gray')
plt.show()
Notch_ft=np.fft.fftshift(notch('btw', 500, 500, 50,20,100,1))
plt.title('Butterworth Notch Filter'), plt.imshow(Notch_ft, cmap = 'gray')
plt.show()
Notch_ft=np.fft.fftshift(notch('gaussian', 500, 500, 50,20,50,1))
plt.title('Gaussian Notch Filter'), plt.imshow(Notch_ft, cmap = 'gray')
plt.show()