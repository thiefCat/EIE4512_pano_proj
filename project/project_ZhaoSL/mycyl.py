import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt

def cylindricalWarping(img, f):
    '''
    input: image to be warped, focal length f
    output: the warpped image
    '''
    h, w = img.shape[:2]
    # print(img.shape)
    cx = w/2
    cy = h/2

    u = np.arange(w)
    v = np.arange(h)[:, None]
    mappedX = f * np.tan((u-cx)/f) + cx + v*0
    mappedY = (v-cy) * (f**2 + ((f * np.tan((u-cx)/f) + cx)-cx)**2)**0.5 / f + cy
    mappedX = mappedX.astype(np.float32)
    mappedY = mappedY.astype(np.float32)

    return cv2.remap(img, mappedX, mappedY, cv2.INTER_CUBIC)

# img1 = cv2.imread('/Users/zhaosonglin/Desktop/IMG_1006.jpeg')
# res = cylindricalWarping(img1, 5700)
# plt.imshow(res)
# plt.show()



