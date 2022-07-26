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
    # for offset in range(L):
    H[int(poz_x), int(poz_y-L):int(poz_y)] = 1
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


def wiener_deblur_color0(blurred, L=35, K=0.1):
    '''deblur color img (wiener filter)'''
    H = motion_process(blurred.shape[:2], L)
    result = np.zeros_like(blurred)
    for i in range(3):
        layer = wiener_filter(blurred[:,:,i], H, K)
        layer[layer > 255] = 255
        result[:,:,i] = layer
    return result


def wiener_deblur_color(blurred, v, T, K, mask):
    '''deblur color img (wiener filter)'''   
    L = int(v * T)
    # print('blur L: ', L)
    if L < 5:
        # if the blur is too smaller than `20` pixels, skip deblur
        print('--skip--')
        return blurred  

    H = motion_process(blurred.shape[:2], L)
    result = np.zeros_like(blurred)
    for i in range(3):
        channel = wiener_filter(blurred[:,:,i], H, K)
        channel[mask==1] = 0
        channel[channel > 255] = 255
        result[:,:,i] = channel
    return result
    

def inverse_deblur_color(blurred, L=35, eps=1e-3):
    '''deblur color img (inverse filter)'''
    H = motion_process(blurred.shape[:2], L)
    result = np.zeros_like(blurred)
    for i in range(3):
        layer = inverse_filter(blurred[:,:,i], H, eps)
        layer[layer > 255] = 255
        result[:,:,i] = layer
    return result

def test():
    img = cv2.imread('test_blur_21_6.png')
    # img = cylindricalWarping(img, 1503)
    spatial_filter = motion_process(img.shape[:2], 30)
    # spatial_filter[spatial_filter!=0] = 255
    cv2.imwrite('filter1.png', np.uint8(spatial_filter))

    res_wie = wiener_deblur_color0(img, L=12, K=0.05)
    res_inv = inverse_deblur_color(img, L=28)

    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.imshow(img)
    plt.subplot(132)
    plt.imshow(res_wie)
    cv2.imwrite('deblurred.png', np.uint8(res_wie))
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


if __name__ == '__main__':
    test()
