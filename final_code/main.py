import image_stitching
import mycyl
from get_f import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
from frame_selector import Frame_selector
import velocity
import motion_deblur

print('-------selecting frames-------')
FS = Frame_selector()
FS.set_path('videos\\7.19_2.MOV')
FS.set_path('videos\\7.21_6.MOV')
# FS.set_path('videos\\7.21_3.MOV')
imgs = FS.run_select_frame(proxy_compress=3,
                           sift_thres=0.5,
                           interest_thres=40)

print('------cylindrical warping------')
f = get_f(24, 3/5, 1080, 3000, 2160)
for i in range(len(imgs)):
    img = imgs[i]
    img = mycyl.cylindricalWarping(img, f)
    imgs[i] = img

print('-----deblurring------')
pattern = np.amax(imgs[0], axis=2)
h, w = pattern.shape[:2]
mask = np.zeros_like(pattern) # construct the cylindrical mask
mask[pattern == 0] = 1

neighbors = FS.out_neighb_frames()

T = 1/35 # exposure time

for i in range(len(imgs)):
    # calculate velocity of selected frames
    print('-frame ',i)
    img = imgs[i]
    imgl = neighbors[i][0]
    imgr = neighbors[i][1]
    v = velocity.get_velocity(imgl, imgr, f, ratio=0.6, t=2/FS.fps)
    print('-velocity: ',v)

    # deblurring with masks
    mask_tmp = mask.copy()
    mask_tmp[:,:int(v/9)] = 1 
    img = motion_deblur.wiener_deblur_color(img, v, T, 0.1, mask_tmp)
    imgs[i] = img

print('-----stitching images------')
stitcher = image_stitching.Stitcher()
res = stitcher.run_stitch_divide(imgs[:], 0.6)
res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
plt.imshow(res)
plt.show()


# save the result
res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
cv2.imwrite('panorama_out.png', np.uint8(res))