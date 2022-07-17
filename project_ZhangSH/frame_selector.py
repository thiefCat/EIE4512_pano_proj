# this file realize selecting frames from a video

import cv2 as cv2
import numpy as np 

class Frame_selector:

    def __init__(self):
        ''' basic'''
        self.path = None             # path of the video
        self.frame_set_proxy = None  # array of all frames ndarray
        self.frame_set_origin = None # frames without compression
        self.L = None                # total number of frames
        self.selected_frames = []    # indexes of interest frames
        self.f = None
        ''' criterion'''
        self.sift_thres = None       # threshold of ratio test
        self.max_length = None       # maxlength of interest points
        self.interest_thres = None   # threshold of number of interest points

    def set_path(self, path=None):
        self.path = path
    
    def set_focal(self, focal=1000):
        self.f = focal

    def set_threshold(self, sift_thres: float, max_length: int, interest_thres):
        self.sift_thres = sift_thres         # threshold of the ratio test
        self.max_length = max_length         # max number of interest point
        self.interest_thres = interest_thres # threshold number of interest point

    def load_vedio(self, proxy_compress=1):
        ''' read video file, return ndarray containing frames'''
        capture = cv2.VideoCapture(self.path) 
        frame_set_proxy  = []
        frame_set_origin = []

        while True:
            isTrue, frame = capture.read()

            if isTrue:
                ''' record the frame_set'''
                h,w = frame.shape[:2]
                frame = cv2.rotate(frame, cv2.ROTATE_180) 
                frame_set_origin.append(frame) # original
                frame_cps = cv2.resize(frame, dsize = (w//proxy_compress, h//proxy_compress), interpolation=cv2.INTER_CUBIC)
                frame_set_proxy.append(frame_cps)    # compressed

            else:
                break

        capture.release()
        cv2.destroyAllWindows()

        self.frame_set_proxy  = np.array(frame_set_proxy)
        self.frame_set_origin = np.array(frame_set_origin)
        self.L                = len(frame_set_proxy)

        return self.frame_set_proxy, self.frame_set_origin, self.L 

    def play_video(self):
        ''' play the video (proxy one)'''        
        for frame in self.frame_set_proxy:
            cv2.imshow('Video', frame)
            if cv2.waitKey(20) & 0xFF==ord('d'):
                break 
    
    def show_frame(self, frame_idx: int, ifshow=True):
        if frame_idx > self.L-1:
            print('ERROR: frame index out of range')
        else:
            cv2.imshow('{} frame'.format(str(frame_idx)), self.frame_set_proxy[frame_idx])
            cv2.waitKey(0)
        return self.frame_set_proxy[frame_idx]


    # criterion for matching ---------------------------------------------

    def __sift_matching(self, img1, img2, threshold: float, max_length: int):
        '''find interest pairs using the SIFT algorithm'''

        sift = cv2.xfeatures2d.SIFT_create()

        keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

        # create matches
        matches = cv2.BFMatcher().knnMatch(descriptors_1, descriptors_2, k = 2) # tuple
        # apply ratio test (Lowe's: nearest/next_nearest)
        good_match = []
        good_indexes = []
        for (m, n) in matches:
            # max_length = 30 # max number of corr pts
            if (m.distance < threshold * n.distance) and (len(good_match) < max_length):
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

    def __get_criterion(self, idx1, idx2):
        '''generate a function of criterion for interest point'''
        return 200 * self.interest_thres * np.log1p((idx2-idx1)) / self.L
        # return self.interest_thres

    def __reach_critirion(self, idx1, idx2):
        ''' check if two frame_set has at least 10 interest pts with small threshold'''
        frame1 = self.frame_set_proxy[idx1]
        frame2 = self.frame_set_proxy[idx2]
        _, good_indexes, _, _ = self.__sift_matching(frame1, frame2, self.sift_thres, self.max_length)
        return len(good_indexes) > self.__get_criterion(idx1, idx2)

    # binary search on frame_set -------------------------------------------------

    def search_frames(self):
        ''' search in all the frames'''
        self._search_frames(1, self.L-2)

    def _search_frames(self, start: int, end: int):
        ''' do binary search on frame_set
        stop search if criterion reached
        (assumption: camera motion is oriented)'''
        self.__search(start, end)

        return self.selected_frames


    def __search(self, idx1, idx2):
        ''' recursive search'''
        if (idx2-idx1 < 2) or self.__reach_critirion(idx1, idx2):
            # criterion reached
            if idx1 == idx2:
                self.selected_frames.append(idx1)
                return idx1, -1

            self.selected_frames.append(idx2)                
            return idx1, idx2

        else:
            print(idx1,idx2)
            self.__search(idx1, (idx2+idx1)//2)
            self.__search((idx2+idx1)//2, idx2)
    
    # output --------------------------------------------

    def print_selected(self):
        ''' print indexes of selected frames'''
        print(self.selected_frames)
        return np.array(self.selected_frames)


    def output_selected_frames(self, show=True, save=False, if_original=False):
        ''' imshow and return selected frames, save if save==True'''
        # choose if output uncompressed img
        frame_set = {True: self.frame_set_origin, 
                     False: self.frame_set_proxy}[if_original]
        
        output_frames = []
        for index in self.selected_frames:
            output_frames.append(frame_set[index])

            if show:
                pass
                cv2.imshow('frame_{}'.format(str(index)), frame_set[index])            
            if save:
                cv2.imwrite('frame_{}.png'.format(str(index)), np.uint8(frame_set[index]))

        cv2.waitKey(0)
        return np.array(output_frames)


## ------------------------------------------------------------

if __name__ == '__main__':

    FF = Frame_selector()
    FF.set_path('video\IMG_4804.MOV')
    FF.set_focal(28) 
    FF.load_vedio(proxy_compress=5)
    FF.set_threshold(sift_thres=0.5, max_length=50, interest_thres=10)
    # FF.play_video()
    # FF.show_frame(100)
    FF.search_frames()
    FF.print_selected()
    frames = FF.output_selected_frames(show=True, if_original=False)
