import image_stitching
import mycyl
import cv2
import matplotlib.pyplot as plt
import numpy as np
from frame_selector import Frame_selector
import velocity
import motion_deblur

print('-------selecting frames-------')
FF = Frame_selector()
FF.set_path('/Users/zhaosonglin/Documents/GitHub/EIE4512_pano_proj/final_code/videos/7.19_2.MOV')
imgs = FF.run_select_frame(proxy_compress=5,
                           sift_thres=0.5,
                           interest_thres=5)



print('------cylindrical warping------')
for i in range(len(imgs)):
    img = imgs[i]
    img = mycyl.cylindricalWarping(img, 1544)
    imgs[i] = img

pattern = np.amax(imgs[0], axis=2)
h = pattern.shape[0]
w = pattern.shape[1]
mask = np.zeros_like(pattern)
mask[pattern == 0] = 1

print('-----deblurring------')
neighbors = FF.out_neighb_frames()
T = 1/30
for i in range(len(imgs)):
    print(i)
    img = imgs[i]
    imgl = neighbors[i][0]
    imgr = neighbors[i][1]
    v = velocity.get_velocity(imgl, imgr, 1544, 0.6, 1/15)
    print(v)
    img = motion_deblur.wiener_deblur_color(img, v, T, 0.1, mask)
    imgs[i] = img



# for img in imgs:
#     cv2.imshow('img', img)
#     cv2.waitKey(0)

# img1 = cv2.imread('/Users/zhaosonglin/Desktop/IMG_1006.jpeg')
# img2 = cv2.imread('/Users/zhaosonglin/Desktop/IMG_1007.jpeg')
# img3 = cv2.imread('/Users/zhaosonglin/Desktop/IMG_1008.jpeg')
# imgs = [img1, img2, img3]

# for (index,img) in enumerate(imgs):

#     f = 6000
#     # cyl, cyl_mask = cylindricalWarpImage(img, K)
#     cyl_ = mycyl.cylindricalWarping(img, f)
#     imgs[index] = cyl_
#     # cv2.imshow('cyl{}'.format(str(index)), cyl)
#     # cv2.imshow('msk', cyl_mask)
#     # cv2.imwrite('cylin_{}.png'.format(str(index)), np.uint8(cyl))
#     # cv2.imwrite('mask_{}.jpg'.format(str(index)), np.uint8(cyl_mask))

print('-----stitching images------')
stitcher = image_stitching.Stitcher()
# res = stitcher.run_stitch_sequencial(imgs, 0.6)
res = stitcher.run_stitch_divide(imgs, 0.6)
# print(res)
res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
plt.imshow(res)
plt.show()