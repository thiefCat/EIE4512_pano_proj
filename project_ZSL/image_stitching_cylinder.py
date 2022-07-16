import cv2
import numpy as np
import matplotlib.pyplot as plt
import queue
import math

class Stitcher():
    def __init__(self):
        pass
    def stitch(self, img1, img2, ratio = 0.3):
        print('creating sift-----')
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        print('matching-----')
        match = cv2.BFMatcher(normType=cv2.NORM_L2)
        matches = match.knnMatch(des1, des2, k=2)
        good_match = []
        for m, n in matches:
            if m.distance < ratio * n.distance:   
                good_match.append(m)
        MIN_MATCH_COUNT = 5
        print('finding homography------')
        print('number of good match:', len(good_match))
        if len(good_match) > MIN_MATCH_COUNT:
            dst_pts = np.float32([ kp1[m.queryIdx].pt for m in good_match ]).reshape(-1,1,2)
            src_pts = np.float32([ kp2[m.trainIdx].pt for m in good_match ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        else:
            print('not enough matches are found:', len(good_match))

        # self.draw(dst_pts, src_pts, good_match)

        print('warping images')
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        c1 = M @ np.array([w2, h2, 1])
        c2 = M @ np.array([w2, 0, 1])
        
        # output image size
        H = max(h1, h2, math.ceil(c1[1]/c1[2]))
        W = math.ceil(max(c1[0]/c1[2], c2[0]/c2[2]))

        dst = cv2.warpPerspective(img2, M, (W, H))
        # dst[0:img1.shape[0], 0:img1.shape[1]] = img1
        dst_ = np.amax(dst, axis=2)  # left size
        # print(dst_.shape)
        mask = np.zeros((H, W))     # left size
        mask[dst_ == 0] = 1    
        img1_= np.zeros((H, W, 3))

        # print(h1, w1)
        # print(img1.shape)
        # print(img1_.shape)

        img1_[:h1, :w1] = img1[:h1, :w1]
        dst[mask==1] = img1_[mask==1]

        return dst

    def run(self, imgs, ratio):
        q = queue.Queue()
        for img in imgs:
            q.put(img)
        size = q.qsize()

        a = q.get()
        b = q.get()
        print('number of stitch:', 1)
        prev = self.stitch(a, b, ratio)

        for i in range(size-2):
            print('number of stitch:', i+2)
            now = q.get()
            prev = self.stitch(prev, now, ratio)
        print('-----complete----')

        return prev