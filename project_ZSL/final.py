import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('/Users/zhaosonglin/Desktop/programming/python/project/project5-panorama-thiefCat-main/data/source005/file0001.jpg')
img2 = cv2.imread('/Users/zhaosonglin/Desktop/programming/python/project/project5-panorama-thiefCat-main/data/source005/file0003.jpg')
# img1_uint8 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX)
# img2_uint8 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
match = cv2.BFMatcher(normType=cv2.NORM_L2)
matches = match.knnMatch(des1, des2, k=2)
good_match = []
for m, n in matches:
    if m.distance < 0.5 * n.distance:   
        good_match.append(m)

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
# out_img2 = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
# out_img2[:h1, :w1] = img1
# out_img2[:h2, w1:w1 + w2] = img2

# out_img2 = cv2.drawMatches(img1, kp1, img2, kp2, good_match, out_img2)

# plt.imshow(out_img2)
# plt.show()

MIN_MATCH_COUNT = 5
if len(good_match) > MIN_MATCH_COUNT:
    dst_pts = np.float32([ kp1[m.queryIdx].pt for m in good_match ]).reshape(-1,1,2)
    src_pts = np.float32([ kp2[m.trainIdx].pt for m in good_match ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

else:
    print('not enough matches are found:', len(good_match))



dst = cv2.warpPerspective(img2, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))
dst[0:img1.shape[0], 0:img1.shape[1]] = img1
plt.imshow(dst)
plt.show()
# sadhfoaishvipqiwewvadsnkd