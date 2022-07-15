import cv2 as cv2
import numpy as np 
import matplotlib.pyplot as plt 

def translation(tx: int, ty: int):
    # return the translational 3*3 matrix
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], np.float32) # translation
    return T

## sift matching -----------------------------------------


def sift_matching(img1, img2, threshold: float, max_length: int):

    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    # create matches
    matches = cv2.BFMatcher().knnMatch(descriptors_1, descriptors_2, k = 2) # tuple
    # print(len(matches))

    # apply ratio test (Lowe's: nearest/next_nearest)
    good_match = []
    good_indexes = []
    for (m, n) in matches:
        
        # max_length = 30 # max number of corr pts

        if (m.distance < threshold * n.distance) and (len(good_match) < max_length):
            good_match.append([m])

            # get corres indexes in ketpts
            index1 = m.queryIdx
            index2 = m.trainIdx
            good_indexes.append((index1, index2))
    
    # draw correspondence onto one img
    img3 = cv2.drawMatchesKnn(img1, keypoints_1, img2, keypoints_2, good_match, None,
                              matchColor=(0, 255, 0), matchesMask=None,
                              singlePointColor=(255, 0, 0), flags=0)

    return (img3, good_indexes, keypoints_1, keypoints_2)


## show corr on original imgs --------------------------------------

def find_corrs_loc(keypoints_1: tuple, keypoints_2: tuple, good_indexes: list) -> list:
    # returns locations of corr pts in two original images
    pts_loc1 = []
    pts_loc2 = []

    for (index1, index2) in good_indexes:
        loc1 = keypoints_1[index1].pt
        loc2 = keypoints_2[index2].pt
        pts_loc1.append(loc1)
        pts_loc2.append(loc2)
    
    return (pts_loc1, pts_loc2)


def mark_pts(img, pts_loc: list):
    # mark a set of pts on img
    img_out = img.copy()
    leng = len(pts_loc)

    for (i, pt) in enumerate(pts_loc):

        color = (0, int(i/leng*255), 255)
        cv2.circle(img_out, (int(pt[0]),int(pt[1])), 4, color, thickness=-1)

    return img_out


## solve homography coefficients -------------------------------

def solve_homogrphy(pts_loc1: list, pts_loc2: list):
    # solve homography mapping matrix H(3*3)

    L = len(pts_loc1)

    if L < 4:
        # under-estimated system
        print('WARNING-The number of points is too small for estimation: {}'.format(str(L)))
        exit()

    else:

        M, p_d = linsys_constructor(pts_loc1, pts_loc2, L)

        if L == 4:
            p_s = np.linalg.solve(M, p_d)
        else:
            # over-estimated system
            M_ = np.matmul(M.T, M)
            p_d_ = np.matmul(M.T, p_d)
            p_s = np.linalg.solve(M_, p_d_)

    H = np.concatenate((p_s.flatten(), [1]))    
    return H.reshape((3,3)).astype(np.float32)


def linsys_constructor(pts_loc1: list, pts_loc2: list, L: int):
    # construct the 2L*8 coefficient matrix M and destination p (L>8: overestimate)
    # 1:x,y/ 2:u,v
    pts1 = np.array(pts_loc1)
    pts2 = np.array(pts_loc2)

    X, Y = pts1[:, 0].reshape(L,1), pts1[:, 1].reshape(L,1)
    U, V = pts2[:, 0].reshape(L,1), pts2[:, 1].reshape(L,1)

    M = np.zeros((2*L,8))

    M[0:L,   0:1] = U
    M[0:L,   1:2] = V
    M[0:L,   2:3] = 1
    M[L:2*L, 3:4] = U
    M[L:2*L, 4:5] = V
    M[L:2*L, 5:6] = 1
    M[0:L,   6:7] = -U * X
    M[0:L,   7:8] = -V * X
    M[L:2*L, 6:7] = -U * Y
    M[L:2*L, 7:8] = -V * Y

    return M.copy(), np.concatenate((X, Y), axis=0)


## apply homography transformation ----------------------

def homo_point(u,v,T):
    # do homography transform on one point

    a,b,c,d,e,f,g,h,i = T.flatten()
    x = (a*u+b*v+c) / (g*u+h*v+i)
    y = (d*u+e*v+f) / (g*u+h*v+i)
    return int(x), int(y)


def homogaphy_trans(img1, T):
    # calculate homography transform of img and mask

    w, h, _ = img1.shape
    print('shape', img1.shape)
    w1, h1 = homo_point(0, 0, T)
    w2, h2 = homo_point(0, h, T)
    w3, h3 = homo_point(w, 0, T)
    w4, h4 = homo_point(w, h, T)
    W = int(max(w1,w2,w3,w4, w))
    H = int(max(h1,h2,h3,h4, h))
    # print(W,H)


    mask = np.ones_like(img1)
    mask_trans = cv2.warpPerspective(mask, T, dsize = (W, H))
    img_trans = cv2.warpPerspective(img1, T, dsize = (W, H))

    return img_trans, mask_trans


def merge_warpped(img1, img2):

    w1, h1, _ = img1.shape
    w2, h2, _ = img2.shape
    w = max(w1, w2)
    h = max(h1, h2)

    img_out = np.zeros((w,h,3), dtype='uint8')

    img1_1 = img_out.copy()
    img2_1 = img_out.copy()
    img1_1[0:w1, 0:h1,:] = img1[0:w1, 0:h1,:]
    img2_1[0:w2, 0:h2,:] = img2[0:w2, 0:h2,:]

    for i in range(w):
        for j in range(h):

            if sum(img2_1[i,j,:]) > sum(img1_1[i,j,:]):
                img_out[i,j,:] = img2_1[i,j,:]
            else:
                img_out[i,j,:] = img1_1[i,j,:]

    return img_out


# encapsulation ------------------------------------------

def construct_pano_2(img1, img2):
    # parameters
    scale = 1
    threshold = 0.15
    max_length = 50
    trans_x = 0
    trans_y = 0

    # do translation of img1
    T = translation(trans_x, trans_y)
    img1, _ = homogaphy_trans(img1, T)

    # cv2.imshow('1t', img1)
    # cv2.waitKey(0)

    # apply SIFT and match algorithm
    img3, good_indexes, keypoints_1, keypoints_2 \
        = sift_matching(img1, img2, threshold, max_length)
        # cv2.imshow('Feature matching', img3)
    print('\t1------Feature matched')

    # find corresponding points
    pts_loc1, pts_loc2 = find_corrs_loc(keypoints_1, keypoints_2, good_indexes)
    print('\t2------Corresponding pts founded')

    # solve homography matrix
    H = solve_homogrphy(pts_loc1, pts_loc2)
    # print('homography matrix:\n', H)
    print('\t3------Homography matrix obtained')

    # apply homogaphy on img2
    img2_trans, _ = homogaphy_trans(img2, H)
    print('\t4------Image wrapped')

    # stitch imgs
    stitched = merge_warpped(img1, img2_trans)
    print('stitch shape: ', stitched.shape)
    # cv2.imshow('stitched', stitched)
    # cv2.waitKey(0)
    print('\t5------Image stitched')

    return stitched


def pano_multiple(imgs: list):
    # construct panorama from multiple images
    leng = len(imgs)
    if leng > 1:
        print('[Start to construct pano]')
        pano = construct_pano_2(imgs[0], imgs[1])
        ptr = 2 
        while ptr < leng:
            print('[image #{}]'.format(str(ptr)))
            pano = construct_pano_2(pano, imgs[ptr])
            # pano = construct_pano_2(imgs[ptr], pano)
            ptr += 1
    return pano


if __name__ == '__main__':
    # input imgs
    img1 = cv2.imread('data\source002\source002_01.jpg') # query img
    img2 = cv2.imread('data\source002\source002_02.jpg') # train img

    pano = pano_multiple([img1, img2])

    cv2.imshow('pano', pano)
    cv2.waitKey(0)




'''
data\source001\source001_01.jpg
data\source001\source001_02.jpg

data\source002\source002_01.jpg
data\source002\source002_02.jpg

data\source003\panorama02_01.jpg
data\source003\panorama02_02.jpg

data\source004\panorama03_03.jpg
data\source004\panorama03_04.jpg
data\source004\panorama03_07.jpg

data\source005\\file0001.jpg
data\source005\\file0002.jpg
data\source005\\file0003.jpg
data\source005\\file0004.jpg
data\source005\\file0005.jpg

data\source006\yosemite1.jpg
data\source006\yosemite2.jpg
data\source006\yosemite3.jpg

data\source007\\0.jpg
data\source007\\1.jpg
data\source007\\2.jpg
data\source007\\3.jpg

data\source008\\01.jpg
data\source008\\02.jpg
data\source008\\03.jpg
'''

