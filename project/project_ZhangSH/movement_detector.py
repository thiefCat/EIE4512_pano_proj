import numpy as np
import cv2 as cv2
import sys

from frame_selector import Frame_selector


frame1 = cv2.imread('frame_1.png')
frame2 = cv2.imread('frame_2.png')


def find_move(frame1, frame2):
    '''find the movement of frame2 w.r.t. frame1'''
    h, w = frame1.shape[:2]
    dist = min(w,h) // 5
    cropped = frame2[h//2-dist:h//2+dist, w//2-dist:w//2+dist]
    
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    h, w = frame1.shape[:2]
    x_move, y_move = 0, 0
    curr_dif = sys.maxsize

    for x in range(-dist, dist):
        for y in range(-dist, +dist):

            # compare the cropped section with a part in original image
            diff= np.abs(
                cropped - frame1[h//2+x-dist:h//2+x+dist, w//2+y-dist:w//2+y+dist]).sum()

            if diff < curr_dif:
                print(diff, 'x=', x, 'y=', y)
                curr_dif = diff
                x_move, y_move = x, y
                # x,y is the point in frame1 that corres to the mid of img2 
                # (on the right of the center in frame1)

    print(x_move, y_move)
    return x_move, y_move

def _find_xy(frame1, crop2, d0:int, x0:int, y0:int, step:int):
    print('new recursion d0----------: ', d0)

    if step <= 2:
        return x0, y0

    else:
        curr_dif = sys.maxsize
        for x in range(x0-step, x0+step, step//3):

            for y in range(y0-step, y0+step, step//3):

                print([x0-d0+x-(x0+d0+x), y0-d0+y-(y0+d0+y)])

                crop1 = frame1[x0-d0+x:x0+d0+x, y0-d0+y:y0+d0+y]
                # crop1 = frame1[:2*d0, :2*d0]
                print('shape1: ',crop1.shape)
                print('shape2: ',crop2.shape)

                # compare the crop1 section with a part in original image
                diff = np.abs(crop1 - crop2).sum()

                if diff < curr_dif:
                    print('find smaller ----------')
                    print(diff, 'x=', x, 'y=', y)
                    curr_dif = diff
                    x_move, y_move = x, y
                    
        _find_xy(frame1, crop2, d0, x0+x_move, y0+y_move, step//3)


def find_move1(frame1, frame2):
    '''find the movement of frame2 w.r.t. frame1'''
    h, w = frame1.shape[:2]
    d0 = min(w,h) // 5
    crop2 = frame2[h//2-d0:h//2+d0, w//2-d0:w//2+d0]
    
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    crop2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)

    h, w = frame1.shape[:2]
    x_move, y_move = 0, 0

    x_move, y_move = _find_xy(frame1, crop2, d0, x0=h//2, y0=w//2, step=d0) # recursive

    return x_move, y_move




xm, ym = find_move(frame1, frame2)
# find_move1(frame1, frame2)


