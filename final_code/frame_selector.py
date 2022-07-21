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
        
        self.motion = []             # translation motion of the selected frames

    def set_path(self, path=None):
        self.path = path
    
    def set_threshold(self, sift_thres: float, interest_thres):
        self.sift_thres = sift_thres         # threshold of the ratio test in SIFT
        self.interest_thres = interest_thres # threshold number related to interest point

    def exposure_time(self):
        if self.fps is not None:
            return 1 / self.fps


    # video methods --------------------------------------------------------------------------------

    def load_vedio(self, proxy_compress=4, rotate=True):
        ''' read video file, return ndarray containing frames'''
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
        self.L                = len(frame_set_proxy)
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


    def __reach_critirion(self, idx1, idx2):
        ''' check if two frame_set has at least 10 interest pts with small threshold'''
        frame1 = self.frames_proxy[idx1]
        frame2 = self.frames_proxy[idx2]
        img3, good_indexes, kps1, kps2 = self.__sift_matching(frame1, frame2, self.sift_thres)
        # two different criterion for matching -------
        reach = len(good_indexes) > (len(kps1) + len(kps2)) / self.interest_thres # 300 initially
        reach1 = len(good_indexes) > 15 * self.interest_thres * np.log1p((idx2-idx1)) / self.L

        return reach1

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
            print('-comparing frames: ', idx1,idx2)
            self.__search(idx1, (idx2+idx1)//2) # check left
            self.__search((idx2+idx1)//2, idx2) # check right
    

    # output --------------------------------------------------------
    
    def imshow_selected(self):
        '''imshow selected frames'''
        for index in self.selected_frames:
            cv2.imshow('frame_{}'.format(str(index)), self.frames_proxy[index])
        cv2.waitKey(0)
        return

    def output_selected_frames(self, save=False, if_original=False):
        ''' imshow and return selected frames, save if save==True'''
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

    FF = Frame_selector()
    FF.set_path('videos\\7.18_8.MOV') 
    # FF.load_vedio(proxy_compress=5)
    # FF.set_threshold(sift_thres=0.5, interest_thres=10)
    ## FF.play_video()
    ## FF.show_frame(100)
    # FF.search_frames()
    # FF.print_selected()
    # frames = FF.output_selected_frames(if_original=False)


    FF.run_select_frame(proxy_compress=3,
                        sift_thres=0.3,
                        interest_thres=200)

    neigb_frames = FF.out_neighb_frames()

'''


