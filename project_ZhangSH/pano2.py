import cv2 as cv2
import numpy as np 
import matplotlib.pyplot as plt 

def translation(tx: int, ty: int):
    # return the translational 3*3 matrix
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], np.float32) # translation
    return T


# T = translation(100,100)
# warpshape1 = (img1.shape[0]*2, img1.shape[1]*2)
# img1 = cv2.warpPerspective(img1, T, dsize = warpshape1)

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
        print('---- The number of points is too small for estimation ----')
        return None

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
    return H.reshape((3,3))


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


def construct_pano_2(img1, img2):

    # do translation of img1
    T = translation(50,50)
    scale = 2
    warpshape1 = (img1.shape[0]*scale, img1.shape[1]*scale)
    img1 = cv2.warpPerspective(img1, T, dsize = warpshape1)

    # apply SIFT and match algorithm
    img3, good_indexes, keypoints_1, keypoints_2 \
        = sift_matching(img1, img2, threshold=0.2, max_length=30)

    # find corresponding points
    pts_loc1, pts_loc2 = find_corrs_loc(keypoints_1, keypoints_2, good_indexes)

    # solve homography matrix
    H = solve_homogrphy(pts_loc1, pts_loc2)
    print('homography matrix:\n', H)

    # apply homogaphy on img2
    warpshape2 = (img2.shape[0]*scale, img2.shape[1]*scale)
    img2_warp = cv2.warpPerspective(img2, H, dsize = warpshape2)

    # stitch imgs
    stitched = merge_warpped(img1, img2_warp)
    cv2.imshow('stitched', stitched)
    cv2.waitKey(0)


if __name__ == '__main__':
    # input imgs
    img1 = cv2.imread('data\\source005\\file0001.jpg') # query img
    img2 = cv2.imread('data\\source005\\file0003.jpg') # train img
    construct_pano_2(img1, img2)

