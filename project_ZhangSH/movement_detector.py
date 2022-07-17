import cv2 as cv2
import numpy as np 

from frame_selector import Frame_selector


FF = Frame_selector()
FF.set_path('video\IMG_4804.MOV')
FF.set_focal(28) 

# proxy, _, length = FF.load_vedio(proxy_compress=5)


img = cv2.imread('data\source001\source001_01.jpg')


frame1 = FF.show_frame(100)
frame2 = FF.show_frame(101)


def gradient_x(img):
    return img[:,1:,:] - img[:,:-1,:]
def gradient_y(img):
    return img[1:,:,:] - img[:-1,:,:]
def second_gradient_x(img):
    return np.abs(img[:,2:,:] - 2 * img[:,1:-1,:] + img[:,:-2,:])
def second_gradient_y(img):
    return np.abs(img[2:,:,:] - 2 * img[1:-1,:,:] + img[:-2,:,:])

def loss_func(u,v,imgs,index):
    img_curr = imgs[index]
    img_nxt = imgs[index+1]
    loss = u * gradient_x(img_curr) + v * gradient_y(img_nxt) + (img_nxt-img_curr)
    return np.sum(loss)

img_x = gradient_x(img)
img_y = gradient_y(img)

cv2.imshow('x gred', img_x)
cv2.imshow('y gred', img_y)
cv2.waitKey(0)