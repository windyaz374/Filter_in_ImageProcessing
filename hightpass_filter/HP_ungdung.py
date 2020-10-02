import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from hpft import hpft
plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)
img = cv.imread('footBall_orig.jpg',0)
img_fft = np.fft.fft2(img)
plt.subplot(151), plt.imshow(np.log(1+np.abs(img_fft)), "gray"), plt.title("Spectrum")
img_fft_center = np.fft.fftshift(img_fft)
plt.subplot(152), plt.imshow(np.log(1+np.abs(img_fft_center)), "gray"), plt.title("Centered Spectrum")
inv_center = np.fft.ifftshift(img_fft_center)
plt.subplot(153), plt.imshow(np.log(1+np.abs(inv_center)), "gray"), plt.title("Decentralized")
processed_img = np.fft.ifft2(inv_center)
plt.subplot(154), plt.imshow(np.abs(processed_img), "gray"), plt.title("Processed Image")
plt.show()

Ideal_HPF = hpft('ideal', img.shape[0], img.shape[1], 1,1)
Ideal_HPF_Center = np.fft.fftshift(Ideal_HPF)
Butterworth_HPF = hpft('btw', img.shape[0], img.shape[1], 1,1)
Butterworth_HPF_Center = np.fft.fftshift(Butterworth_HPF)
Gauss_HPF = hpft('gaussian',img.shape[0], img.shape[1], 1,1)
Gauss_HPF_Center = np.fft.fftshift(Gauss_HPF)

Ideal_Center = Ideal_HPF_Center*img_fft_center
plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)
plt.subplot(152), plt.imshow(np.log(1+np.abs(Ideal_Center)), "gray"), plt.title("Spectrum of image with ideal highpass filter")
output = np.fft.ifft2(Ideal_Center)
plt.subplot(154), plt.imshow((np.abs(output)), "gray"), plt.title("Image with ideal highpass filter")
plt.show()

Butterworth_Center = Butterworth_HPF_Center*img_fft_center
plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)
plt.subplot(152), plt.imshow(np.log(1+np.abs(Butterworth_Center)), "gray"), plt.title("Spectrum of image with Butterworth highpass filter")
output = np.fft.ifft2(Butterworth_Center)
plt.subplot(154), plt.imshow((np.abs(output)), "gray"), plt.title("Image with Butterworth highpass filter")
plt.show()

Gaussian_Center = Gauss_HPF_Center*img_fft_center
plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)
plt.subplot(152), plt.imshow(np.log(1+np.abs(Gaussian_Center)), "gray"), plt.title("Spectrum of image with Gaussian highpass filter")
output = np.fft.ifft2(Gaussian_Center)
plt.subplot(154), plt.imshow((np.abs(output)), "gray"), plt.title("Image with Gaussian highpass filter")
plt.show()