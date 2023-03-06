
# import cv2 as cv2
# import numpy as np 
# import cvxpy as cp
# from scipy.optimize import minimize, linprog

# from frame_selector import Frame_selector

# FF = Frame_selector()
# FF.set_path('video\IMG_4804.MOV')
# FF.load_vedio(proxy_compress=5)
# frame1 = FF.show_frame(100)
# frame2 = FF.show_frame(120)
# cv2.imwrite('frame_1.png', np.uint8(frame1))
# cv2.imwrite('frame_20.png', np.uint8(frame2))

# frame1 = cv2.imread('frame_1.png')
# frame2 = cv2.imread('frame_2.png')

# frame1 = cv2.imread('data\source001\source001_01.jpg') 
# frame2 = cv2.rotate(frame1, cv2.ROTATE_180)
# frame2 = frame1[10:,:]
# frame1 = frame1[:-10,:]


# def get_mid(img, depth):
#     h, w = img.shape[:2]
#     return img[h//2-depth : h//2+depth, w//2-depth : w//2+depth]

# depth = 20
# center = get_mid(frame2, depth)

# # cv2.imshow('frame_1.png', frame1)
# # cv2.imshow('frame_2.png', frame2)
# # cv2.waitKey(0)


# imgs = [frame1, frame2]

# def calc_gred_x(img):
#     h, w = img.shape[:2]
#     kernel_x = np.array([[-1,-1,-1], [0,0,0], [1,1,1]]) # Prewitt
#     padded = np.pad(img, (1,1,), 'edge')
#     dst_ = cv2.filter2D(padded, ddepth = -1, kernel = kernel_x)
#     dst_ = dst_[1:h+1, 1:w+1, 1:4]
#     return dst_

# def calc_gred_y(img):
#     h, w = img.shape[:2]
#     kernel_y = np.array([[-1,0,1], [-1,0,1], [-1,0,1]]) # Prewitt
#     padded = np.pad(img, (1,1,), 'edge')
#     dst_ = cv2.filter2D(padded, ddepth = -1, kernel = kernel_y)
#     dst_ = dst_[1:h+1, 1:w+1, 1:4]
#     return dst_


# def loss_func(imgs, index=0):
#     '''objective function'''
#     img_cur = imgs[index]
#     img_nxt = imgs[index+1]
#     h, w = img_cur.shape[:2]

#     gred_x = calc_gred_x(img_cur)
#     print(gred_x)
#     gred_y = calc_gred_y(img_cur)
#     opt_flow = img_nxt - img_cur
#     cv2.imshow('gred_x', gred_x)
#     cv2.imshow('gred_y', gred_y)
#     cv2.imshow('optical flow', opt_flow)
#     cv2.waitKey(0)
#     def val(x):
#         '''objective value for specific frames'''   
#         # loss = (x[0]*gred_x + x[1]*gred_y + (opt_flow))**2
#         loss = (x[0]*gred_x + x[1]*gred_y + opt_flow)**2

#         print(loss.sum())
#         return loss.sum()

#     return val

# def solve_loss_opt(imgs):

#     D0 = np.array([0,0])
#     args = [imgs, 0]

#     cons = ({'type': 'ineq', 'fun': lambda x: x[0]-10},
#             {'type': 'ineq', 'fun': lambda x: x[1]-10})

#     res = minimize(loss_func(args[0], args[1]), D0, method='nelder-mead', constraints=cons)
#     return res

# res = solve_loss_opt(imgs)
# print(res)
# print(res.fun)
# print(res.success)
# print(res.x)

# # cv2.waitKey(0)




