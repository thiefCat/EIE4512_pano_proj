import image_stitching
import mycyl
from get_f import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
from frame_selector import Frame_selector


print('-------selecting frames-------')
FF = Frame_selector()
FF.set_path('videos\\7.19_1.MOV')
imgs = FF.run_select_frame(proxy_compress=3,
                        sift_thres=0.5,
                        interest_thres=10)

print('------cylindrical warping------')
f = get_f(26, 3/5, 1080, 3000, 2160)
for i in range(len(imgs)):
    img = imgs[i]
    img = mycyl.cylindricalWarping(img, f)
    imgs[i] = img

print('-----warping images------')
stitcher = image_stitching.Stitcher()
res = stitcher.run_stitch_divide(imgs, 0.6)
res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
plt.imshow(res)
plt.show()