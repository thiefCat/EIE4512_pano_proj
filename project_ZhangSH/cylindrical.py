# this file implements transformations (Cylindrical and projective)

import numpy as np
import cv2 as cv2
import math

from image_stitching_copy import Stitcher


## homography transformation -------------------------------------

def __homo_point(u,v,M):
    # do homography transform on one point 
    # u/x: h, v/y: w
    a,b,c,d,e,f,g,h,i = M.flatten()
    x = (a*u+b*v+c) / (g*u+h*v+i)
    y = (d*u+e*v+f) / (g*u+h*v+i)
    return int(x), int(y)


def projective_warp(img, M):
    # projective warping of colored image

    h, w = img.shape[:2]
    h1, w1 = __homo_point(0, 0, M)
    h2, w2 = __homo_point(0, w, M)
    h3, w3 = __homo_point(h, 0, M)
    h4, w4 = __homo_point(h, w, M)
    W = int(max(w, w1, w2, w3, w4))
    H = int(max(h, h1, h2, h3, h4))

    img_warpped = cv2.warpPerspective(img, M, dsize=(W, H))
    # mask = __projective_warp_mask(img, M)

    # return img_warpped, mask
    return img_warpped


def __projective_warp_mask(img, K, erosion=0):
    # returns the mask of cylindrical mapping
    msk_before = np.full_like(img, 255)
    mask = projective_warp(msk_before, K)
    if erosion > 0:
        kernel = np.ones((3,3))
        mask = cv2.erode(mask, kernel)
    return mask

## cylindrical transformation ---------------------------------------

def cylindrical_warp(img, K):

    """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
    h_, w_ = img.shape[:2]
    # pixel coordinates
    y_i, x_i = np.indices((h_,w_))
    X = np.stack([x_i,y_i,np.ones_like(x_i)], axis=-1).reshape(h_*w_,3) # to homog
    Kinv = np.linalg.inv(K) 
    X = Kinv.dot(X.T).T # normalized coords
    # calculate cylindrical coords (sin\theta, h, cos\theta)
    A = np.stack([np.sin(X[:,0]), X[:,1], np.cos(X[:,0])],axis=-1).reshape(w_*h_,3)
    B = K.dot(A.T).T # project back to image-pixels plane
    # back from homog coords
    B = B[:,:-1] / B[:,[-1]]
    # make sure warp coords only within image bounds
    B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
    B = B.reshape(h_, w_, -1)

    img_rgba = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA) # for transparent borders...
    # warp the image according to cylindrical coords

    img_cyl = cv2.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
    
    # mask = __cylindrical_warp_mask(img, M)

    # return img_cyl, mask
    return img_cyl


def __cylindrical_warp_mask(img, K, erosion=0):
    # returns the mask of cylindrical mapping
    msk_before = np.full_like(img, 255)
    mask = cylindrical_warp(msk_before, K)
    if erosion > 0:
        kernel = np.ones((3,3))
        mask = cv2.erode(mask, kernel)
    return mask

## ------------------------------------------------------------

if __name__ == '__main__':

    imgs = []
    masks = []
    imgs.append(cv2.imread('pano0\\frame_42.jpg'))
    imgs.append(cv2.imread('pano0\\frame_63.jpg'))
    imgs.append(cv2.imread('pano0\\frame_85.jpg'))
    imgs.append(cv2.imread('pano0\\frame_106.jpg'))
    imgs.append(cv2.imread('pano0\\frame_128.jpg'))
    imgs.append(cv2.imread('pano0\\frame_149.jpg'))
    imgs.append(cv2.imread('pano0\\frame_171.jpg'))

    for (index,img) in enumerate(imgs):

        h, w = img.shape[:2]
        K = np.array([[700,0,w/2],
                    [0,700,h/2],
                    [0,0,1]]) # mock intrinsics

        M = np.array([[1, 0.03, 0],
                    [0, 1, 0],
                    [0, 0.00015, 1]])

        cyl = cylindrical_warp(img, K)
        cyl_mask = __cylindrical_warp_mask(img, K)

        cyl = projective_warp(cyl, M)
        cyl_mask = projective_warp(cyl_mask, M)

        imgs[index] = cyl
        masks.append(cyl_mask)

        cv2.imshow('cyl{}'.format(str(index)), cyl)
        cv2.imshow('msk', cyl_mask)

        # cv2.imwrite('cylin_{}.png'.format(str(index)), np.uint8(cyl))
        # cv2.imwrite('mask_{}.jpg'.format(str(index)), np.uint8(cyl_mask))

    cv2.waitKey(0)