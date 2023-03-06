# this file implements cylindrical transformations 

import numpy as np
import cv2 as cv2
import math

class Transformer:

    def __init__(self, imgs, f):
        '''inits with a set of imgs and camers focal val'''

        self.L = len(imgs)                  # number of images

        self.imgs = imgs                    # input images
        self.h, self.w = imgs[0].shape[:2]  # set height and width
        self.f = f                          # set focal

        self.M = np.array([[1,0,0],[0,1,0],[0,0,1]])     # matrix for homography transform

        self.K = np.array([ [self.f, 0,      self.w/2], 
                            [0,      self.f, self.h/2], 
                            [0,      0,      1       ]]) # matrix for cylindrical transform
                            # this variable only for square pixels

        self.cyl_imgs =    [None] * self.L
        self.cyl_masks =   [None] * self.L
        self.trans_imgs =  [None] * self.L
        self.trans_masks = [None] * self.L
        

    def set_M(self, M):
        '''set the homography transformation matrix'''
        self.M = M
        

    ## homography transformation -------------------------------------

    def __homo_point(self, u,v,M):
        ''' do homography transform on one point 
        u/x: h, v/y: w'''
        a,b,c,d,e,f,g,h,i = M.flatten()
        x = (a*u+b*v+c) / (g*u+h*v+i)
        y = (d*u+e*v+f) / (g*u+h*v+i)
        return int(x), int(y)


    def add_homography(self):
        '''do homography to all imgs'''

        for index in range(self.L):
            cyl = self.cyl_imgs[index]
            msk = self.cyl_masks[index]
            trans_img = self._projective_warp(cyl, self.M)
            trans_msk = self._projective_warp(msk, self.M)
            self.trans_imgs[index] =  trans_img
            self.trans_masks[index] = trans_msk

        return self.trans_imgs, self.trans_masks


    def _projective_warp(self, img, M):
        ''' projective warping of colored image'''

        h, w = img.shape[:2]
        h1, w1 = self.__homo_point(0, 0, M)
        h2, w2 = self.__homo_point(0, w, M)
        h3, w3 = self.__homo_point(h, 0, M)
        h4, w4 = self.__homo_point(h, w, M)
        W = int(max(w, w1, w2, w3, w4))
        H = int(max(h, h1, h2, h3, h4))

        img_warpped = cv2.warpPerspective(img, M, dsize=(W, H))
        # mask = __projective_warp_mask(img, M)
        # return img_warpped, mask
        return img_warpped


    def __projective_warp_mask(self, img, K, erosion=0):
        ''' returns the mask of cylindrical mapping'''
        msk_before = np.full_like(img, 255)
        mask = self.projective_warp(msk_before, K)
        if erosion > 0:
            kernel = np.ones((3,3))
            mask = cv2.erode(mask, kernel)
        return mask

    ## cylindrical transformation ---------------------------------------

    def construct_cylindricals(self):
        '''construct cylindrical transformations for all imgs'''

        for index in range(self.L):
            img = self.imgs[index]         
            cyl      = self._cylindrical_warp(img)       # do cylin on img
            cyl_mask = self.__cylindrical_warp_mask(img) # do mask on cylin
            self.cyl_imgs[index] = cyl
            self.cyl_masks[index]      = cyl_mask

        return self.cyl_imgs, self.cyl_masks


    def _cylindrical_warp(self, img):
        """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
        K = self.K
        h_, w_ = img.shape[:2]
        # pixel coordinates
        y_i, x_i = np.indices((h_, w_))
        X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(h_*w_, 3) # to homog
        Kinv = np.linalg.inv(K) 
        X = Kinv.dot(X.T).T # normalized coords
        # calculate cylindrical coords (sin\theta, h, cos\theta)
        A = np.stack([np.sin(X[:,0]), X[:,1], np.cos(X[:,0])],axis=-1).reshape(w_*h_, 3)
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


    def __cylindrical_warp_mask(self, img, erosion=0):
        ''' returns the mask of cylindrical mapping'''
        K = self.K
        msk_before = np.full_like(img, 255)
        mask = self._cylindrical_warp(msk_before)
        if erosion > 0:
            kernel = np.ones((3, 3))
            mask = cv2.erode(mask, kernel)
        return mask


    # output ---------------------------------------------

    def output_cyl(self, msk = True, save = False):
        '''imshow final imgs/ type_: cyl & msk'''

        for i, cyl in enumerate(self.trans_imgs):
            cv2.imshow('cyl_{}'.format(str(i)), cyl)
            if msk:
                cv2.imshow('msk_{}'.format(str(i)), self.trans_masks[i])
            
            if save:
                cv2.imwrite('cyl_{}.png'.format(str(i)), np.uint8(cyl))
                if msk:
                    cv2.imwrite('msk_{}.png'.format(str(i)), np.uint8(self.trans_masks[i]))
        cv2.waitKey(0)



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



    TT = Transformer(imgs, f = 800)
    TT.set_M(np.array([[1, 0.04, 0],  [0, 1, 0],  [0, 0.0002, 1]]))
    TT.construct_cylindricals()
    TT.add_homography()
    TT.output_cyl(msk=False)

