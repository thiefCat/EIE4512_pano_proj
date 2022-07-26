import cv2 as cv2
import numpy as np 
import matplotlib.pyplot as plt
import math
from numpy import fft
from mycyl import *


def motion_process(img_shape, L):
    '''simulate motion blur'''
    H = np.zeros(img_shape)
    poz_x = (img_shape[0] - 1) / 2
    poz_y = (img_shape[1] - 1) / 2    
    for offset in range(L):
        H[int(poz_x), int(poz_y-offset)] = 1
    return H / H.sum()


def make_blurred(input, H, eps):
    '''motion blur img'''
    input_fft = fft.fft2(input)
    H_fft = fft.fft2(H) + eps
    blurred = fft.ifft2(input_fft * H_fft)
    blurred = np.abs(fft.fftshift(blurred))
    return blurred

def inverse_filter(input, H, eps):  
    '''Inverse filtering'''
    input_fft = fft.fft2(input)
    H_fft = fft.fft2(H) + eps
    result = fft.ifft2(input_fft / H_fft)  
    result = np.abs(fft.fftshift(result))
    return result
    
def wiener_filter(img, H, K):
    '''filter the blurred img'''
    H /= np.sum(H)
    result = np.copy(img)
    result = fft.fft2(result)
    H = fft.fft2(H, s=img.shape)
    H = np.conj(H) / (np.abs(H) ** 2 + K)
    result = result * H
    result = np.abs(fft.ifft2(result))
    result = fft.ifftshift(result)
    return result


def wiener_deblur_color(blurred, v, T, K, mask):
    '''deblur color img (wiener filter)'''
    L = int(v * T)
    H = motion_process(blurred.shape[:2], L)
    result = np.zeros_like(blurred)
    channel1 = wiener_filter(blurred[:,:,0], H, K)
    channel1[mask==1] = 0
    channel1[channel1 > 255] = 255
    channel2 = wiener_filter(blurred[:,:,1], H, K)
    channel2[mask==1] = 0
    channel2[channel2 > 255] = 255
    channel3 = wiener_filter(blurred[:,:,2], H, K)
    channel3[mask==1] = 0
    channel3[channel3 > 255] = 255
    result[:,:,0] = channel1
    result[:,:,1] = channel2
    result[:,:,2] = channel3
    return result

def inverse_deblur_color(blurred, L=35, eps=1e-3):
    '''deblur color img (inverse filter)'''
    H = motion_process(blurred.shape[:2], L)
    result = np.zeros_like(blurred)
    result[:,:,0] = inverse_filter(blurred[:,:,0], H, eps)
    result[:,:,1] = inverse_filter(blurred[:,:,1], H, eps)
    result[:,:,2] = inverse_filter(blurred[:,:,2], H, eps)
    return result

def test():
    img = cv2.imread('frame_50.png')
    # img = cylindricalWarping(img, 1503)
    res_wie = wiener_deblur_color(img, L=28, K=0.05)
    res_inv = inverse_deblur_color(img, L=28)

    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.imshow(img)
    plt.subplot(132)
    plt.imshow(res_wie)
    plt.subplot(133)
    plt.imshow(res_inv)
    plt.show()

    # plt.figure(figsize=(10,20))
    # plt.gray()
    
    # blurred = cv2.imread('frame_50.png', 0)
    # H = motion_process(blurred.shape[:2], 35)
    # blurred = cylindricalWarping(blurred, 1503)

    # plt.subplot(131), plt.xlabel("Original blurred"), plt.imshow(blurred)
 
    # result_w = wiener_filter(blurred, H, 0.05) 
    # plt.subplot(132), plt.xlabel("Wiener deblurred"), plt.imshow(result_w)

    # result_i = inverse_filter(blurred, H, 0.1+1e-3) 
    # plt.subplot(133), plt.xlabel("Inverse deblurred"), plt.imshow(result_i)

    # plt.show()


# if __name__ == '__main__':
#     test()
