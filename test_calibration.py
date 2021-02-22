#!/usr/bin/env python
import cv2
import numpy as np
import os
import glob
if __name__ == '__main__':
    filepath = 'video.flv'

    npzfile = np.load('calibr_parameters.npz')
    mtxstr, diststr = npzfile.files
    mtx = npzfile[mtxstr]
    dist = npzfile[diststr]


    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        print("VideoCapture failed...")
    else:
        print("Video opened! ...")    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS:", fps)

    ret, frame = cap.read()         # Read first frame to get info about the resolution
    h,w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))
    x,y,w,h = roi
    out = cv2.VideoWriter('undistorted.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (w, h))

    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    dst = dst[y:y+h, x:x+w]
    out.write(dst)

    while cap.isOpened():
        ret, frame = cap.read()
        if np.shape(frame) == ():
            continue
        
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        dst = dst[y:y+h, x:x+w]
        out.write(dst)
        #cv2.imshow("video", dst)
        #k = cv2.waitKey(30) & 0xff                   # if 'Esc' is pressed then quit
        #if k == 27:
        #    break

    cv2.destroyAllWindows()
    out.release()
    cap.release()
