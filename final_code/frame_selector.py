# this file realize selecting frames from a video

import cv2 as cv2
import numpy as np 

class Frame_selector:

    # settings ----------------------------------------------------------------------

    def __init__(self):
        ''' basic'''
        self.path = ''               # path of the video
        self.frames_proxy = []       # frames with compression
        self.frames_origin = []      # frames with no compression
        self.L = None                # total number of frames
        self.fps = None              # frame rate
        self.selected_frames = []    # indexes of interest frames
        self.video_size = (0,0)      # height and width of video

        ''' criterion'''
        self.sift_thres = None       # threshold of ratio test
        self.interest_thres = None   # threshold of number of interest points

    def set_path(self, path=None):
        self.path = path
    
    def set_threshold(self, sift_thres: float, interest_thres):
        self.sift_thres = sift_thres         # threshold of the ratio test in SIFT
        self.interest_thres = interest_thres # threshold number related to interest point

    def exposure_time(self):
        if self.fps is not None:
            return 1 / self.fps

    def add_all_frames_to_selected_frames(self):
        for idx in range(1,self.L-1):
            self.selected_frames.append(idx) 
        return self.selected_frames

    # video methods --------------------------------------------------------------------------------

    def load_vedio(self, proxy_compress=4, rotate=False):
        ''' read video file, return lists containing frames'''
        print('-start loading video...')
        capture = cv2.VideoCapture(self.path) 
        self.fps = capture.get(cv2.CAP_PROP_FPS)
        print('-fps:', self.fps)
        # print('read success:', capture.isOpened())
        frame_set_proxy  = []
        frame_set_origin = []

        while True:
            isTrue, frame = capture.read()

            if isTrue:
                ''' record the frame_set'''
                h,w = frame.shape[:2]
                if rotate:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                frame_set_origin.append(frame)       # original
                frame_cps = cv2.resize(frame, dsize = (w//proxy_compress, h//proxy_compress), interpolation=cv2.INTER_CUBIC)
                frame_set_proxy.append(frame_cps)    # compressed

            else:
                break

        capture.release()
        cv2.destroyAllWindows()

        self.frames_proxy  = np.array(frame_set_proxy)
        self.frames_origin = np.array(frame_set_origin)
        self.L             = len(frame_set_proxy)
        print('-video length: {} frames'.format(self.L))

        return self.frames_proxy, self.frames_origin, self.L 

    def play_video(self):
        ''' play the video (proxy one)'''        
        for frame in self.frames_proxy:
            cv2.imshow('Video', frame)
            if cv2.waitKey(20) & 0xFF==ord('d'):
                break 
    
    def show_frame(self, frame_idx: int, ifshow=False):
        if frame_idx > self.L-1:
            print('ERROR: frame index out of range')
        elif ifshow == True:
            cv2.imshow('{} frame'.format(str(frame_idx)), self.frames_proxy[frame_idx])
            cv2.waitKey(0)
        return self.frames_proxy[frame_idx]


    # selecting frames ---------------------------------------------

    def out_sift_matching(self, img1, img2, threshold: float):
        return self.__sift_matching(img1, img2, threshold)

    def __sift_matching(self, img1, img2, threshold: float):
        '''find interest pairs using the SIFT algorithm'''

        sift = cv2.SIFT_create()

        keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

        # create matches
        matches = cv2.BFMatcher().knnMatch(descriptors_1, descriptors_2, k = 2) # tuple

        # apply ratio test (Lowe's: nearest/next_nearest)
        good_match = []
        good_indexes = []

        for (m, n) in matches:
            if (m.distance < threshold * n.distance):
                good_match.append([m])

                # get corres indexes in ketpts
                index1 = m.queryIdx
                index2 = m.trainIdx
                good_indexes.append((index1, index2))
        
        # draw correspondence onto one img
        img3 = cv2.drawMatchesKnn(img1, keypoints_1, img2, keypoints_2, good_match, None,
                                matchColor=(0, 255, 0), matchesMask=None,
                                singlePointColor=(255, 0, 0), flags=0)

        return (img3, good_indexes, keypoints_1, keypoints_2)


    def __reach_criterion(self, idx1, idx2):
        ''' check if two frame_set has at least 10 interest pts with small threshold'''
        frame1 = self.frames_proxy[idx1]
        frame2 = self.frames_proxy[idx2]
        img3, good_indexes, kps1, kps2 = self.__sift_matching(frame1, frame2, self.sift_thres)
        # cv2.imwrite('match_demo_{}-{}.png'.format(idx1,idx2), np.uint8(img3))
        # different criterion for matching -------
        reach = len(good_indexes) > (len(kps1) + len(kps2)) / self.interest_thres
        reach1 = len(good_indexes) > 15 * self.interest_thres * np.log1p((idx2-idx1)) / self.L
        reach2 = len(good_indexes) > 6 * self.interest_thres * (idx2-idx1) / self.L
       
        return reach2

    # binary search on frame_set -------------------------------------------------

    def search_frames(self):
        ''' search in all the frames'''
        self._search_frames(1, self.L-2)

    def _search_frames(self, start: int, end: int):
        ''' do binary search on frame_set
        stop search if criterion reached
        (assumption: camera motion is oriented)'''
        self.selected_frames.append(start)
        self.selected_frames.append(end)

        self.__search(start, end)

        self.selected_frames = sorted(self.selected_frames)

        return self.selected_frames


    def __search(self, idx1, idx2):
        ''' recursive search'''
        
        if ((idx2-idx1 < 2) or self.__reach_criterion(idx1, idx2))\
            and (idx2-idx1) < self.L//2:
            # criterion reached 
            return              

        else:
            self.selected_frames.append((idx2+idx1)//2)
            print('-comparing frames: ', idx1,idx2)
            self.__search(idx1, (idx2+idx1)//2) # check left
            self.__search((idx2+idx1)//2, idx2) # check right
            
    # def __search(self, idx1, idx2):
    #     ''' recursive search'''
    #     if (idx2-idx1 < 2) or self.__reach_criterion(idx1, idx2):
    #         # criterion reached
    #         if idx1 == idx2:
    #             self.selected_frames.append(idx1)
    #             return idx1, -1

    #         self.selected_frames.append(idx2)                
    #         return idx1, idx2

    #     else:
    #         print('-comparing frames: ', idx1,idx2)
    #         self.__search(idx1, (idx2+idx1)//2) # check left
    #         self.__search((idx2+idx1)//2, idx2) # check right
    

    # output --------------------------------------------------------

    def output_selected_frames(self, save=False, if_original=False):
        ''' imshow and return selected frames, save to disk if save==True'''
        # choose if output uncompressed img
        frame_set = {True: self.frames_origin, 
                     False: self.frames_proxy}[if_original]
        
        output_frames = []
        for index in self.selected_frames:
            output_frames.append(frame_set[index])         
            if save:
                cv2.imwrite('frame_{}.png'.format(str(index)), np.uint8(frame_set[index]))
        return output_frames # type = list
    
    def run_select_frame(self, if_original=False, proxy_compress=5, sift_thres=0.5, interest_thres=10):
        '''main method'''
        self.load_vedio(proxy_compress)
        self.set_threshold(sift_thres, interest_thres)
        self.search_frames()
        res = self.output_selected_frames(if_original=True)
        print('length of the imgs:', len(self.selected_frames))
        print('selcted frames:', self.selected_frames)
        return res
    
    def out_neighb_frames(self):
        '''output neighboring frames of the selected frames'''
        neigb = []
        for index in self.selected_frames:
            frame1 = self.frames_origin[index-1]
            frame2 = self.frames_origin[index+1]
            # cv2.imshow('frame1', frame1)
            # cv2.imshow('frame2', frame2)
            # cv2.waitKey(0)
            neigb.append((frame1, frame2))
        return neigb


## -----------------------------------------------------------------

'''
if __name__ == '__main__':

    FS = Frame_selector()
    FS.set_path('videos\\7.19_3.MOV') 
    # FS.load_vedio(proxy_compress=5)
    # FS.set_threshold(sift_thres=0.5, interest_thres=10)
    ## FS.play_video()
    ## FS.show_frame(100)
    # FS.search_frames()
    # FS.print_selected()
    # frames = FS.output_selected_frames(if_original=False)


    FS.run_select_frame(proxy_compress=3,
                        sift_thres=0.5,
                        interest_thres=30)

    # neigb_frames = FS.out_neighb_frames()

    FS.output_selected_frames(save=True)

    # img1, img2 = frames[1], frames[2]
    # (img3, good_indexes, keypoints_1, keypoints_2) = FS.__sift_matching(img1, img2, threshold=0.5)
    # cv2.imshow(img3)
    # cv2.imwrite('match_demo_0.png', np.uint8(img3))
    # cv2.waitKey(0)
'''





