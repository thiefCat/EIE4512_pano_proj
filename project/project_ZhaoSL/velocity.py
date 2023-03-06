import cv2
import numpy as np
import math

def get_velocity(img1, img2, f, ratio, t):
    '''
    f: focal length
    ratio: the ratio for the ratio test
    t: time interval between frame_i-1 and frame_i+1
    '''
    
    h, w = img1.shape[:2]
    cx = w/2
    l0 = cx + f * math.atan((0-cx)/f)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    match = cv2.BFMatcher(normType=cv2.NORM_L2)
    matches = match.knnMatch(des1, des2, k=2)
    good_match = []
    for m, n in matches:
        if m.distance < ratio * n.distance:   
            good_match.append(m)
    MIN_MATCH_COUNT = 5
    if len(good_match) >= MIN_MATCH_COUNT:
        dst_pts = np.float32([ kp1[m.queryIdx].pt for m in good_match ]).reshape(-1,1,2)
        src_pts = np.float32([ kp2[m.trainIdx].pt for m in good_match ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    else:
        print('not enough matches are found:', len(good_match))

    h2, w2 = img2.shape[:2]
    c1 = M @ np.array([w2, h2, 1])    # right bottom point after transformation
    c2 = M @ np.array([w2, 0, 1])     # right top point after transformation

    W2 = (c1[0]/c1[2] + c2[0]/c2[2])/2 - 2*l0
    W1 = w - 2*l0
    v = (W2-W1)/t
    return v


img1 = cv2.imread('/Users/zhaosonglin/Documents/GitHub/EIE4512_pano_proj/source_cyl_warp_f400/cylin_0.jpg')
img2 = cv2.imread('/Users/zhaosonglin/Documents/GitHub/EIE4512_pano_proj/source_cyl_warp_f400/cylin_1.jpg')
# get_velocity(img1, img2, 1544, 0.6, )
