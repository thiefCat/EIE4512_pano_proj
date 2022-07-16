import image_stitching
import cyl
import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread('/Users/zhaosonglin/Desktop/IMG_1006.jpeg')
img2 = cv2.imread('/Users/zhaosonglin/Desktop/IMG_1007.jpeg')
img3 = cv2.imread('/Users/zhaosonglin/Desktop/IMG_1008.jpeg')
imgs = [img1, img2, img3]
for (index,img) in enumerate(imgs):

    h, w = img.shape[:2]
    K = np.array([[8000,0,w/2],[0,8000,h/2],[0,0,1]]) # mock intrinsics
    # cyl, cyl_mask = cylindricalWarpImage(img, K)
    cyl_ = cyl.cylindricalWarp(img, K)
    cyl_ = cv2.cvtColor(cyl_, cv2.COLOR_BGR2RGB)
    cyl_ = cv2.cvtColor(cyl_, cv2.COLOR_RGB2BGR)
    imgs[index] = cyl_

    # cv2.imshow('cyl{}'.format(str(index)), cyl)
    # cv2.imshow('msk', cyl_mask)

    # cv2.imwrite('cylin_{}.png'.format(str(index)), np.uint8(cyl))
    # cv2.imwrite('mask_{}.jpg'.format(str(index)), np.uint8(cyl_mask))


stitcher = image_stitching.Stitcher()
res = stitcher.run(imgs, 0.5)
res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
plt.imshow(res)
plt.show()

