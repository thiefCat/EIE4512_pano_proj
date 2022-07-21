import cv2 as cv2
import numpy as np 
import matplotlib.pyplot as plt
import math
from numpy import fft
from mycyl import *


# Simulate motion blur
def motion_process(img_shape, L):
    PSF = np.zeros(img_shape)
    print(img_shape)
    position_x = (img_shape[0] - 1) / 2
    position_y = (img_shape[1] - 1) / 2

    for offset in range(L):
        PSF[int(position_x + 0), int(position_y - offset)] = 1
    return PSF / PSF.sum()  # Normalize brightness to point spread function

# Motion blur the picture
def make_blurred(input, PSF, eps):
    input_fft = fft.fft2(input)  # Take Fourier Transform of a 2D Array
    PSF_fft = fft.fft2(PSF) + eps
    blurred = fft.ifft2(input_fft * PSF_fft)
    blurred = np.abs(fft.fftshift(blurred))
    return blurred


def inverse(input, PSF, eps):  # Inverse filtering
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps  # noise power, which is known, considering epsilon
    result = fft.ifft2(input_fft / PSF_fft)  # Compute the inverse Fourier transform of F(u,v)
    result = np.abs(fft.fftshift(result))
    return result
    
def wiener_filter(img, H, K):
    H /= np.sum(H)
    result = np.copy(img)
    result = fft.fft2(result)
    H = fft.fft2(H, s = img.shape)
    H = np.conj(H) / (np.abs(H) ** 2 + K)
    result = result * H
    result = np.abs(fft.ifft2(result))
    result = fft.ifftshift(result)
    return result

def main():
    image = cv2.imread('frame_3.png', 0)
    img_h = image.shape[0]
    img_w = image.shape[1]
    plt.figure(1)
    plt.xlabel("Original Image")
    plt.gray()
    plt.imshow(image)  # Show original image

    plt.figure(2)
    plt.gray()

    # Do motion blur
    PSF = motion_process((img_h, img_w), 35)
    blurred = np.abs(make_blurred(image, PSF, 1e-3))
    blurred = cv2.imread('frame_50.png', 0)
    blurred = cylindricalWarping(blurred, 1503)
    # blurred = cv2.GaussianBlur(blurred, ksize = (3,3), sigmaX = 2, sigmaY = 2)

    plt.subplot(141), plt.xlabel("Motion blurred"), plt.imshow(blurred)
    # Inverse filtering
    # result = inverse(blurred, PSF, 1e-3) 
    result = wiener_filter(blurred, PSF, 0.1) 
    plt.subplot(142), plt.xlabel("wiener deblurred"), plt.imshow(result)

    # Adding noise, standard_normal produces a random function
    blurred_noisy = blurred + 0.2 * blurred.std() * \
                    np.random.standard_normal(blurred.shape)  

    # Displays an image with added noise and motion blur
    plt.subplot(143), plt.xlabel("motion & noisy blurred"), plt.imshow(blurred_noisy)  

    # Inverse filtering of an image with added noise
    result = inverse(blurred_noisy, PSF, 0.1 + 1e-3)  
    plt.subplot(144), plt.xlabel("inverse deblurred"), plt.imshow(result)

    plt.show()


if __name__ == '__main__':
    main()

# blur = motion_process((1000,1000), 60)
# f_blur = fft.fft2(blur)
# plt.figure(figsize=(10,10))
# plt.subplot(221)
# plt.imshow(blur, cmap='gray')
# plt.subplot(222)
# plt.imshow(np.log1p(np.abs(f_blur)), cmap='gray')
# plt.show()