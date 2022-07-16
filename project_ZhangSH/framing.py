import cv2 as cv2
import numpy as np 

from pano3 import sift_matching
import image_stitching
from cylindrical import cylindricalWarpImage


class Frame_selector:

    def __init__(self):
        ''' basic'''
        self.path = None            # path of the video
        self.frame_set = None       # array of all frames ndarray
        self.frame_num = None       # total number of frames
        self.selected_frames = []   # indexes of interest frames
        self.f = None
        ''' criterion'''
        self.threshold = None       # threshold of ratio test
        self.max_length = None      # maxlength of interest points
        self.interest_num = None    # threshold of number of interest points

    def set_path(self, path):
        self.path = path
    
    def set_focal(self, focal):
        self.f = focal
    
    def set_threshold(self, threshold: float, max_length: int, interest_num):
        self.threshold = threshold
        self.max_length = max_length
        self.interest_num = interest_num

    def read_video(self, compress=5):
        ''' read video file, return ndarray containing frames'''
        capture = cv2.VideoCapture(self.path) 
        frame_set = []

        while True:
            isTrue, frame = capture.read()
            
            # if cv.waitKey(20) & 0xFF==ord('d'):
            # This is the preferred way - if `isTrue` is false (the frame could 
            # not be read, or we're at the end of the video), we immediately
            # break from the loop. 

            if isTrue:
                ''' record the frame_set'''
                w,h,_ = frame.shape
                frame = cv2.resize(frame, dsize = (h//compress, w//compress), interpolation=cv2.INTER_CUBIC)
                frame = cv2.rotate(frame, cv2.ROTATE_180) 
                frame_set.append(frame)
          
            else:
                break

        capture.release()
        cv2.destroyAllWindows()

        self.frame_set = np.array(frame_set)
        self.frame_num = len(frame_set)

        return np.array(frame_set), len(frame_set)

    def play_video(self):
        ''' play the video'''
        for frame in self.frame_set:
            cv2.imshow('Video', frame)
            if cv2.waitKey(20) & 0xFF==ord('d'):
                break 
    
    def display_frame(self, frame_idx: int):
        if frame_idx > self.frame_num-1:
            print('ERROR: frame index out of range')
        else:
            cv2.imshow('{} frame'.format(str(frame_idx)), self.frame_set[frame_idx])
            cv2.waitKey(0)

    # binary search on frame_set -------------------------------------------------

    def __reach_critirion(self, frame1, frame2):
        ''' check if two frame_set has at least 10 interest pts with small threshold'''
        _, good_indexes, _, _ = sift_matching(frame1, frame2, self.threshold, self.max_length)
        return len(good_indexes) > self.interest_num

    def search_frames(self):
        ''' search in all the frames'''
        self._search_frames(0, self.frame_num-1)

    def _search_frames(self, start: int, end: int):
        ''' do binary search on frame_set
        stop search if criterion reached
        (assumption: camera motion is oriented)'''
        self.__search(start, end)

        return self.selected_frames


    def __search(self, idx1, idx2):
        ''' recursive search'''
        if (idx2-idx1 < 2) or self.__reach_critirion(self.frame_set[idx1], self.frame_set[idx2]):
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

    def show_selected_frames(self):
        ''' imshow selected frames'''
        for index in self.selected_frames:
            cv2.imshow('Frame {}'.format(str(index)), self.frame_set[index])
        cv2.waitKey(0)

    def output_selected_frames(self):
        ''' returns selected frames'''
        frames = []
        for index in self.selected_frames:
            frames.append(self.frame_set[index])
        return np.array(frames)
    
    def save_frames(self):
        ''' save selected frames into computer'''
        for index in self.selected_frames:
            cv2.imwrite('frame_{}.jpg'.format(str(index)), np.uint8(self.frame_set[index]))
            

V = Frame_selector()
V.set_path('video\IMG_4804.MOV')
V.set_focal(28)
V.read_video()
V.set_threshold(0.5, 50, 10)
# V.play_video()
# V.display_frame(100)
V.search_frames()
V.print_selected()
V.show_selected_frames()
# V.save_frames()

frames = V.output_selected_frames()

# stitcher = image_stitching.Stitcher()
# stitcher.run(frames, 0.8)

# print(frames.shape)