import cv2 as cv2
import numpy as np 
from scipy.optimize import minimize

from frame_selector import Frame_selector

# FF = Frame_selector()
# FF.set_path('video\IMG_4804.MOV')
# FF.load_vedio(proxy_compress=5)
# frame1 = FF.show_frame(100)
# frame2 = FF.show_frame(101)
# frame3 = FF.show_frame(102)
# frame4 = FF.show_frame(103)
# frame20 = FF.show_frame(120)
# cv2.imwrite('frame_1.png', np.uint8(frame1))
# cv2.imwrite('frame_2.png', np.uint8(frame2))
# cv2.imwrite('frame_3.png', np.uint8(frame3))
# cv2.imwrite('frame_4.png', np.uint8(frame4))
# cv2.imwrite('frame_20.png', np.uint8(frame20))


# frame1 = cv2.imread('test_motion_rec\IMG_4862.JPG') # o
# frame2 = cv2.imread('test_motion_rec\IMG_4863.JPG') # r [-1.72224348  0.26868462]
# frame3 = cv2.imread('test_motion_rec\IMG_4864.JPG') # u [-1.57923728  0.24837391]
# frame4 = cv2.imread('test_motion_rec\IMG_4865.JPG') # d [-1.39219948  0.21832606]
# frame5 = cv2.imread('test_motion_rec\IMG_4866.JPG') # l [-1.46757246  0.23058398]

frame1 = cv2.imread('frame_1.png')
frame2 = cv2.imread('frame_2.png')
# h,w = frame1.shape[:2]
# frame2 = frame1[1:,1:]
# frame3 = frame1[:-1,:-1]
# frame4 = frame1[:h-1,1:]
# frame5 = frame1[1:,:w-1]




def get_mid(img):
    h, w = img.shape[:2]
    return img[h//2-20:h//2+20, w//2-20:w//2+20]

# frame1 = get_mid(frame1)
# frame2 = get_mid(frame2)

cv2.imshow('frame_1.png', frame1)
cv2.imshow('frame_2.png', frame2)
cv2.waitKey(0)

imgs = [frame1, frame2]


# def loss_func(imgs, index=0):
#     '''objective function'''
#     img_cur = imgs[index]
#     img_nxt = imgs[index+1]

#     intensity_cur = img_cur.min(2)
#     intensity_nxt = img_nxt.min(2)
#     gred_x = ((intensity_cur[:,1:] - intensity_cur[:,:-1])[1:,:])
#     gred_y = ((intensity_cur[1:,:] - intensity_cur[:-1,:])[:,1:])
#     opt_flow = ((intensity_nxt - intensity_cur)[1:,1:])
#     # cv2.imshow('gred_x', gred_x)
#     # cv2.imshow('gred_y', gred_y)
#     # cv2.imshow('optical flow', opt_flow)
#     # cv2.waitKey(0)
#     def val(x):
#         '''objective value for specific frames'''   
#         # loss = (x[0]*gred_x + x[1]*gred_y + (opt_flow))**2
#         loss = ((x[0]*gred_x + x[1]*gred_y + (opt_flow) ).sum())**2

#         print(loss.sum())
#         return loss.sum()

#     return val

def loss_func(imgs, index=0):
    img_cur = imgs[index]
    img_nxt = imgs[index+1]

def solve_loss_opt(imgs):

    D0 = np.array([0,0])
    args = [imgs, 0]
    cons = ()
    res = minimize(loss_func(args[0], args[1]), D0, method='nelder-mead', constraints=cons)
    return res

res = solve_loss_opt(imgs)
print(res)
print(res.fun)
print(res.success)
print(res.x)


