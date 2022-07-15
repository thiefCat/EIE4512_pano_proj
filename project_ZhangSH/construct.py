import os as os
import cv2 as cv2
import numpy as np 

from pano3 import pano_multiple

sources = ['source001', 'source002', 'source003', 'source004', 'source005', 
        'source006', 'source007', 'source008']

DIR = r'data'

for source in sources:

    print('\n'+source)

    path = os.path.join(DIR, source)

    imgs = []

    for (i, img) in enumerate(os.listdir(path)):
        # load imgs in one source
        img_path = os.path.join(path, img)
        
        img_array = cv2.imread(img_path)
        imgs.append(img_array)

        if img_array is None:
            continue

    print('Number of images: {}'.format(str(len(imgs))))
    pano = pano_multiple(imgs)
    cv2.imshow('pano-'+source, pano)
    cv2.waitKey(0)

    cv2.imwrite('pano_{}.jpg'.format(source), np.uint8(pano))










