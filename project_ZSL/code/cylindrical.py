import numpy as np
import cv2 as cv2
import math


'''
Warp an image from cartesian coordinates (x, y) into cylindrical coordinates (theta, h)
Returns: (image, mask)
Mask is [0,255], and has 255s wherever the cylindrical images has a valid value.
Masks are useful for stitching
Usage example:
    im = cv2.imread("myimage.jpg",0) #grayscale
    h,w = im.shape
    f = 700
    K = np.array([[f, 0, w/2], 
                  [0, f, h/2], 
                  [0, 0, 1  ]]) # mock calibration matrix
    imcyl = cylindricalWarpImage(im, K)
'''
def cylindricalWarpImage(img1, K):
    f = K[0,0]

    im_h,im_w,_ = img1.shape

    # go inverse from cylindrical coord to the image
    # (this way there are no gaps)
    cyl = np.zeros_like(img1)
    cyl_mask = np.zeros_like(img1)
    cyl_h,cyl_w,_ = cyl.shape
    x_c = float(cyl_w) / 2.0
    y_c = float(cyl_h) / 2.0

    for x_cyl in np.arange(0,cyl_w):
        for y_cyl in np.arange(0,cyl_h):

            theta = (x_cyl - x_c) / f
            h     = (y_cyl - y_c) / f

            X = np.array([math.sin(theta), h, math.cos(theta)])  # world coordinate
            X = np.dot(K,X)      # image coordinate
            x_im = X[0] / X[2]
            if x_im < 0 or x_im >= im_w:
                continue

            y_im = X[1] / X[2]
            if y_im < 0 or y_im >= im_h:
                continue

            cyl[int(y_cyl), int(x_cyl)] = img1[int(y_im), int(x_im)]
            cyl_mask[int(y_cyl), int(x_cyl)] = 255

    return (cyl,cyl_mask)


img1 = cv2.imread('/Users/zhaosonglin/Desktop/IMG_1006.jpeg')
h, w = img1.shape[:2]
K = np.array([[1200,0,w/2],[0,1200,h/2],[0,0,1]])
res = cylindricalWarpImage(img1, K)
print(res)


# img = cv2.imread('data\source001\source001_01.jpg', 0)

# fs = [20,40,60,80,100,120,140,160,180,200,220]
# for f in fs:
#     h,w = img.shape
#     # f = 200
#     K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])

#     cyl, cyl_mask = cylindricalWarpImage(img, K)
#     cv2.imshow('cyl{}'.format(str(f)), cyl)
#     cv2.imshow('msk', cyl_mask)

#     cv2.waitKey(0)






# import cv2
# import numpy as np


# def cylindricalWarp(img, K):

#     """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
#     h_,w_ = img.shape[:2]
#     # pixel coordinates
#     y_i, x_i = np.indices((h_,w_))
#     X = np.stack([x_i,y_i,np.ones_like(x_i)],axis=-1).reshape(h_*w_,3) # to homog
#     Kinv = np.linalg.inv(K) 
#     X = Kinv.dot(X.T).T # normalized coords
#     # calculate cylindrical coords (sin\theta, h, cos\theta)
#     A = np.stack([np.sin(X[:,0]),X[:,1],np.cos(X[:,0])],axis=-1).reshape(w_*h_,3)
#     B = K.dot(A.T).T # project back to image-pixels plane
#     # back from homog coords
#     B = B[:,:-1] / B[:,[-1]]
#     # make sure warp coords only within image bounds
#     B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
#     B = B.reshape(h_,w_,-1)

#     img_rgba = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA) # for transparent borders...
#     # warp the image according to cylindrical coords
#     return cv2.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)
  


# if __name__ == '__main__':

#     # img = cv2.imread('pano0\\frame_42.jpg')
#     # h, w = img.shape[:2]
#     # K = np.array([[400,0,w/2],[0,400,h/2],[0,0,1]]) # mock intrinsics
#     # img_cyl = cylindricalWarp(img, K)
#     # # cv2.imwrite("image_cyl.png", img_cyl)

#     # cv2.imshow('cylinder', img_cyl)
#     # cv2.waitKey(0)

#     imgs = []
#     masks = []
#     imgs.append(cv2.imread('pano0\\frame_42.jpg'))
#     imgs.append(cv2.imread('pano0\\frame_63.jpg'))
#     imgs.append(cv2.imread('pano0\\frame_85.jpg'))
#     imgs.append(cv2.imread('pano0\\frame_106.jpg'))
#     imgs.append(cv2.imread('pano0\\frame_128.jpg'))
#     imgs.append(cv2.imread('pano0\\frame_149.jpg'))
#     imgs.append(cv2.imread('pano0\\frame_171.jpg'))

#     for (index,img) in enumerate(imgs):

#         h, w = img.shape[:2]
#         K = np.array([[400,0,w/2],[0,400,h/2],[0,0,1]]) # mock intrinsics
#         # cyl, cyl_mask = cylindricalWarpImage(img, K)
#         cyl = cylindricalWarp(img, K)
#         imgs[index] = cyl

#         cv2.imshow('cyl{}'.format(str(index)), cyl)
#         # cv2.imshow('msk', cyl_mask)

#         cv2.imwrite('cylin_{}.png'.format(str(index)), np.uint8(cyl))
#         # cv2.imwrite('mask_{}.jpg'.format(str(index)), np.uint8(cyl_mask))

#     cv2.waitKey(0)

#     stitcher = image_stitching.Stitcher()
#     stitcher.run(imgs, 0.8)
