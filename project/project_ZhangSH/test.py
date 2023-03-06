
import numpy as np
import cv2 as cv2
import math
import sys

from frame_selector import Frame_selector


# FF = Frame_selector()
# FF.set_path('video\IMG_4804.MOV')
# FF.load_vedio(proxy_compress=6)
# frame1 = FF.show_frame(10)
# frame2 = FF.show_frame(16)

frame1 = cv2.imread('frame_1.png')
frame2 = cv2.imread('frame_2.png')


h, w = frame1.shape[:2]


# cv2.imshow('frame1', frame1)
# cv2.waitKey(0)

dist = min(w,h) // 5


def img_crop(img, dist):
    h, w = img.shape[:2]
    ans = img[h//2-dist:h//2+dist, w//2-dist:w//2+dist]
    return ans

crop2 = img_crop(frame2, dist)

# print('h w: ', h,w)


def movement(img1, cropped, dist):

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    h, w = img1.shape[:2]
    xm, ym = -1, -1
    curr_dif = sys.maxsize

    for x in range(h//2-dist, h//2+dist):
        for y in range(w//2-dist, w//2+dist):

            # compare the cropped section with a part in original image
            diff= np.abs(
                cropped - img1[x-dist:x+dist, y-dist:y+dist]).sum()

            if diff < curr_dif:
                print(diff, x, y)
                curr_dif = diff
                # min_xy = [x-h//2,y-w//2] 
                xm, ym = x, y
                # x,y is the point in img1 that corres to the mid of img2 (on the right of the center in img1)
    
    return h//2-xm, w//2-ym


def find_move(frame1, frame2):
    h, w = frame1.shape[:2]
    dist = min(w,h) // 5
    cropped2 = img_crop(frame1, dist)
    x_move, y_move = movement(frame1, cropped2, dist)
    print(x_move, y_move)
    return x_move, y_move


find_move(frame1, frame2)

# x, y = movement(frame1, crop2)
# frame1[x-dist:x+dist, y-dist:y+dist] = img_crop(cv2.Laplacian(frame2, ddepth=-1, ksize=1), dist)


x_move, y_move = movement(frame1, crop2, dist)
print(x_move,y_move)
frame1[h//2-dist-x_move:h//2+dist-x_move, w//2-dist-y_move:w//2+dist-y_move] = img_crop(cv2.Laplacian(frame2, ddepth=-1, ksize=1), dist)

cv2.imshow('effect', frame1)
cv2.waitKey(0)
