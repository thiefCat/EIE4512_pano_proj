import cv2 as cv2
import numpy as np 

from pano3 import sift_matching

def read_video(path):
    # read video file, return ndarray containing frames

    capture = cv2.VideoCapture(path) 
    # print(capture)

    frame_set = []

    while True:
        isTrue, frame = capture.read()
        
        # if cv.waitKey(20) & 0xFF==ord('d'):
        # This is the preferred way - if `isTrue` is false (the frame could 
        # not be read, or we're at the end of the video), we immediately
        # break from the loop. 

        if isTrue:

            # record the frame_set
            w,h,_ = frame.shape
            frame = cv2.resize(frame, dsize = (h//5, w//5), interpolation=cv2.INTER_CUBIC)
            frame = cv2.rotate(frame, cv2.ROTATE_180) 
            frame_set.append(frame)

            '''
            # play the video
            cv2.imshow('Video', frame)
            if cv2.waitKey(20) & 0xFF==ord('d'):
                break 
            '''           
        else:
            break

    capture.release()
    cv2.destroyAllWindows()

    return np.array(frame_set), len(frame_set)

# input video frame_set
path = 'video\IMG_4804.MOV'
frame_set, frame_num = read_video(path)


# print(frame_set[100])
cv2.imshow('100th frame', frame_set[100])
cv2.waitKey(0)


# binary search on frame_set -------------------------------------------------

def reach_critirion(frame1, frame2):
    # check if two frame_set has at least 10 interest pts with small threshold
    _, good_indexes, _, _ = sift_matching(frame1, frame2, 0.15, 300)
    return len(good_indexes) > 10


def search_frames(frame_set, frame_num: int, selected_frames):
    # do binary search on frame_set
    # stop search if criterion reached
    # (assumption: camera motion is oriented)
    
     # containing indexes
    # global selected_frames

    search(0, frame_num-1, frame_set, selected_frames)

    return selected_frames


def search(idx1, idx2, frame_set, selected_frames):

    if (idx2-idx1 < 2) or reach_critirion(frame_set[idx1], frame_set[idx2]):
        selected_frames.append(idx1)
        selected_frames.append(idx2)
        return idx1, idx2
    else:
        search(idx1, (idx2-idx1)//2, frame_set, selected_frames)
        search((idx2-idx1)//2, idx2, frame_set, selected_frames)

selected_frames = []
selected_frames = search_frames(frame_set, frame_num, selected_frames)