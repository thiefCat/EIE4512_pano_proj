import image_stitching
import mycyl
import cv2
import matplotlib.pyplot as plt
import numpy as np
from frame_selector import Frame_selector

print('-------selecting frames-------')
FF = Frame_selector()
FF.set_path('videos\\7.19_6.MOV')
imgs = FF.run_select_frame(proxy_compress=3,
                        sift_thres=0.3,
                        interest_thres=200)

print('------cylindrical warping------')
for i in range(len(imgs)):
    img = imgs[i]
    img = mycyl.cylindricalWarping(img, 1544)
    imgs[i] = img

print('-----warping images------')
stitcher = image_stitching.Stitcher()
res = stitcher.run_stitch_divide(imgs, 0.6)
res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
plt.imshow(res)
plt.show()