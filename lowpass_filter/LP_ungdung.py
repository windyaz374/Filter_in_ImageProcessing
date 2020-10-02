import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from lpft import lpft
plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)
img = cv.imread('fruit2.png',0)
img_fft = np.fft.fft2(img)
plt.subplot(151), plt.imshow(np.log(1+np.abs(img_fft)), "gray"), plt.title("Spectrum")
img_fft_center = np.fft.fftshift(img_fft)
plt.subplot(152), plt.imshow(np.log(1+np.abs(img_fft_center)), "gray"), plt.title("Centered Spectrum")
inv_center = np.fft.ifftshift(img_fft_center)
plt.subplot(153), plt.imshow(np.log(1+np.abs(inv_center)), "gray"), plt.title("Decentralized")
processed_img = np.fft.ifft2(inv_center)
plt.subplot(154), plt.imshow(np.abs(processed_img), "gray"), plt.title("Processed Image")
plt.show()

Ideal_LPF = lpft('ideal', img.shape[0], img.shape[1], 30,5)
Ideal_LPF_Center = np.fft.fftshift(Ideal_LPF)
Butterworth_LPF = lpft('btw', img.shape[0], img.shape[1], 30,5)
Butterworth_LPF_Center = np.fft.fftshift(Butterworth_LPF)
Gauss_LPF = lpft('gaussian',img.shape[0], img.shape[1], 30,5)
Gauss_LPF_Center = np.fft.fftshift(Gauss_LPF)

Ideal_Center = Ideal_LPF_Center*img_fft_center
plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)
plt.subplot(152), plt.imshow(np.log(1+np.abs(Ideal_Center)), "gray"), plt.title("Spectrum of image with ideal lowpass filter")
output = np.fft.ifft2(Ideal_Center)
plt.subplot(154), plt.imshow((np.abs(output)), "gray"), plt.title("Image with ideal lowpass filter")
plt.show()

Butterworth_Center = Butterworth_LPF_Center*img_fft_center
plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)
plt.subplot(152), plt.imshow(np.log(1+np.abs(Butterworth_Center)), "gray"), plt.title("Spectrum of image with Butterworth lowpass filter")
output = np.fft.ifft2(Butterworth_Center)
plt.subplot(154), plt.imshow((np.abs(output)), "gray"), plt.title("Image with Butterworth lowpass filter")
plt.show()

Gaussian_Center = Gauss_LPF_Center*img_fft_center
plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)
plt.subplot(152), plt.imshow(np.log(1+np.abs(Gaussian_Center)), "gray"), plt.title("Spectrum of image with Gaussian lowpass filter")
output = np.fft.ifft2(Gaussian_Center)
plt.subplot(154), plt.imshow((np.abs(output)), "gray"), plt.title("Image with Gaussian lowpass filter")
plt.show()