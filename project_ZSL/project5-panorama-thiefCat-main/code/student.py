import cv2
import scipy.spatial
import numpy as np
import matplotlib.pyplot as plt


def find_correspondences(imgA, imgB, block_size=40):
    ''' 
    Automatically computes a set of correspondence points between two images.

    imgA:         input image A
    imgB:         input image B
    block_size:   size of the area around an interest point that we will 
                  use to create a feature vector. Default to 

    pointsA:      xy locations of the correspondence points in image A
    pointsB:      xy locations of the correspondence points in image B   
    '''

    # Step 1:   Use Harris Corner Detector to find a list of interest points in both images, 
    #   and compute features descriptors for each of those keypoints. Here, we are calculating
    #   the robust SIFT (i.e. Scale Invariant Feature Transform) descriptors, which detects corners
    #   at multiple scales.

    kp1, des1 = find_keypoints_and_features(imgA, block_size)
    kp2, des2 = find_keypoints_and_features(imgB, block_size)

    # Step 2: Find correspondences between the interest points in both images using the feature
    #   descriptors we've calculated for each of the points. 
    #
    # - TODO Step 2a: Calculate and store the distance between feature vectors of all pairs (one from A and one from B) 
    #   of interest points. 
    #   As you may recall, there are many possible distance/similarity metrics. You're welcome to experiment 
    #   but we recommend the L2 norm, tried and true. (hint: scipy.spatial.distance.cdist)

    distances = ???

    # - TODO Step 2b: Find the best matches (pairs of points with the most similarity) that are below some error threshold. 
    #   You're aiming for some number of matches greater than MIN_NUMBER_MATCHES, otherwise you may not have enough information
    #   for later steps. 

    MIN_NUMBER_MATCHES = 20
    RATIO_THRESHOLD = 0.2

    # - TODO: Sort the distances along B's dimension

    # - TODO: For each keypoint, compute the ratio of the top two best matches. This is for Lowe's Ratio test: https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work

    # - TODO: Find the indices of A's keypoints that are below RATIO_THRESHOLD. If there are less than MIN_NUMBER_MATCHES, take the best MIN_NUMBER_MATCHES.
    
    # - TODO: Find the indicies of B's keypoints that belong to A's best matches. And return two arrays containing the best corresponding matches, pointsA and pointsB

    return pointsA, pointsB



def calculate_transform(pointsA, pointsB):
    '''
    This function computes a homography that transforms image A into image B. Using RANSAC, we iteratively
    attempt to compute a matrix that satisfies some random subset of the correspondences we've computed, i.e.
    pointsA and pointsB. RANSAC determines which pairs of corresponding points are inliers, and can be trusted
    in our calculation of the fundamental matrix, and which are outliers. You're welcome to implement this yourself
    for extra credit.
    '''
    M, _ = cv2.findHomography(pointsA, pointsB, cv2.RANSAC, 5.0)

    return M


def warp_images(A, B, transform_M):
    '''
    input:
        imgA, imgB, and transform_M – the 3x3 matrix homography transforming A into B 
                        calculated from their correspondences in previous step 

    returns:        
        warped_A:      image A warped into coordinate space of image B by transform
        warped_B:      image B warped by translation if necessary to keep A in bounds

        These two images will be the same size and they should include the entirety of both images after transformation. 
        This is basically aligning them as necessary to be composited.
    '''
    
    # TODO: Step 1 - Find the bounding box of transformed/warped A in the coordinate frame of B
    #   so that we can determine the dimensions of our composited image.
    A_rect = ??? #Coordinates of the rectangle defining A
    warped_A_rect = cv2.perspectiveTransform(A_rect, transform_M) 

    # TODO: Step 2 - Calculate the translation, if any, that is needed to bring A into fully nonnegative coordinates. 
    #   If we transform A without regard to the bounds, it may get cropped. 
    translation_xy = ??? 

    # TODO: Step 3 - Calculate the width and height of the output image.
    W = ???
    H = ???

    # TODO: Create a translation transform T that translates B to account for any shift of A. This is a 2x3 affine matrix representing the translation.
    transform_T = ???

    # Update transform M with the translation needed to keep A in frame.
    transform_M = np.concatenate((transform_T, [[0, 0, 1]]), axis=0) @ transform_M


    # WARP SPEED
    warped_A = cv2.warpPerspective(A, transform_M, (int(W),int(H)))
    warped_B = cv2.warpAffine(B, transform_T, (int(W),int(H)))

    return warped_A, warped_B

def composite(imgA, imgB):
    '''
    Composite imgA and imgB, both of which have already been warped by warp_images
    '''
    assert(imgA.shape == imgB.shape)
    
    out = np.zeros(imgA.shape, dtype=np.float32)  # placeholder

    return out

def find_keypoints_and_features(img, block_size):
    '''
    input:
        img:            input image
        block_size:     parameter to define block_size x block_size neighborhood around 
                        each pixel used in deciding whether it's a corner

    returns:        
        keypoints:      an array of xy coordinates of interest points
        features:       an array of features corresponding to each of the keypoints
    '''

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_uint8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8') #https://stackoverflow.com/questions/50298329/error-5-image-is-empty-or-has-incorrect-depth-cv-8u-in-function-cvsift

    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(image_uint8,None)

    kp = [[point.pt[0], point.pt[1]] for point in kp]   # coordinates of all the keypoints

    return kp, des