'''
Project 5: Panorama - main.py
CS 1290 Computational Photography, Brown U.
Converted to Python by Megan Gessner.


Usage-

To run on all data:
    
    python main.py

'''

import os
import numpy as np
import argparse
import cv2
from student import find_correspondences, warp_images, composite, calculate_transform
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte


SOURCE_PATH = '../data'
OUTPUT_PATH = '../results'


if __name__ == '__main__':

    # list source directories
    source_dirs = [os.path.join(SOURCE_PATH, f'source00{i+1}') for i in range(len(os.listdir(SOURCE_PATH)))]

    # iterate through each source directory and create panorama from the set of source files in that directory
    for sd, source_dir in enumerate(source_dirs):
        source_files = [os.path.join(source_dir, f'{source_file}') for source_file in sorted(os.listdir(source_dir))]

        A = cv2.imread(source_files[0]) #get the first image
        # A = A.astype(np.float32) / 255. #floats range 0 - 1

        #iterate through the rest of the images
        for source_file in source_files[1:]:

            B = cv2.imread(source_file)
            # B = B.astype(np.float32) / 255.

            # Step 1: Find point correspondences between A and B
            pointsA, pointsB = find_correspondences(A, B)

            # Step 2: Compute homography transforming B to A using RANSAC
            M = calculate_transform(pointsA, pointsB)

            # Step 3: Warp A and B (note: B only translates to account for any negative translation of A in M)
            warped_A, warped_B = warp_images(A, B, M)
            #warped_B, warped_A = warp_images(B, A, np.linalg.inv(M)) # inverse line depending on which direction you want to warp... you may get better results A to B versus B to A

            # Step 4: Stitch the newly-aligned, warped images together
            stitched = composite(warped_A, warped_B)

            # plt.imshow(stitched / 255.0), plt.show()

            # Step 5: A becomes the input to the next loop, if there are more than 2 images in this directory
            A = img_as_ubyte(stitched / 255.0) 

        cv2.imwrite(os.path.join(OUTPUT_PATH, 'panorama_'+str(sd)+'.png'), A)




