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
        prev = self.stitch(a, b, ratio)

        for i in range(size-2):
            print(i)
            now = q.get()
            prev = self.stitch(prev, now, ratio)

        return prev

    # def draw(self, kp1, kp2, good_match):
    #     h1, w1 = img1.shape[:2]
    #     h2, w2 = img2.shape[:2]
    #     out_img2 = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    #     out_img2[:h1, :w1] = img1
    #     out_img2[:h2, w1:w1 + w2] = img2

    #     out_img2 = cv2.drawMatches(img1, kp1, img2, kp2, good_match, out_img2)

    #     plt.imshow(out_img2)
    #     plt.show()

# img1 = cv2.imread('/Users/zhaosonglin/Desktop/IMG_0912.jpeg')
# img2 = cv2.imread('/Users/zhaosonglin/Desktop/IMG_0913.jpeg')
img3 = cv2.imread('/Users/zhaosonglin/Desktop/frames/frame_85.jpg')
img4 = cv2.imread('/Users/zhaosonglin/Desktop/frames/frame_106.jpg')
# img5 = cv2.imread('/Users/zhaosonglin/Desktop/frames/frame_128.jpg')
# img6 = cv2.imread('/Users/zhaosonglin/Desktop/frames/frame_149.jpg')
# img7 = cv2.imread('/Users/zhaosonglin/Desktop/frames/frame_171.jpg')
imgs = [img3, img4]
# print(img3.shape)   # 768, 432
# print(img4.shape)
stitcher = Stitcher()
res = stitcher.run(imgs, 0.5)
# res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
# a = stitcher.stitch(img1, img2, 0.3)
# b = stitcher.stitch(a, img3, 0.3)
# print(a.shape)


plt.imshow(res)
plt.show()

# img_dir = '/Users/zhaosonglin/Desktop/programming/python/project/project5-panorama-thiefCat-main/data/source008'
# names = os.listdir(img_dir)

# imgs = []
# for name in names:
#     img_path = os.path.join(img_dir, name)
#     image = cv2.imread(img_path)
#     imgs.append(image)

# stitcher = Stitcher()
# res = stitcher.run(imgs, 0.3)
# res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
# plt.imshow(res)
# plt.show()

