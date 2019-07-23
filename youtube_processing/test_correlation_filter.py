#!/usr/bin/env python2
import os
import sys
import np
import math
import glob
import cv2
import shutil
from scipy.signal import correlate

PATH_miniclips = "videos_captions/test/"
PARAM_CORR2D_COEFF = 0.79

list_videos = sorted(glob.glob(PATH_miniclips+"*.mp4"), key=os.path.getmtime)
    
for video in list_videos:
    vidcap = cv2.VideoCapture(video)


    if (vidcap.isOpened()== False):
        continue 
        #vidcap.open(video)
    
    corr_list = []
    video_name = video.split("/")[-1]
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_nb_1 in range(0, length - 100, 100):
        vidcap.set(1,frame_nb_1)
        success, image = vidcap.read()
        if success == False:
            continue
        image1 = cv2.resize(image, (100, 50)) 
        #gray_image_1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #for frame_nb_2 in range(frame_nb_1 + 100, length, 100):
        frame_nb_2 = frame_nb_1 + 100
        vidcap.set(1,frame_nb_2)
        success, image = vidcap.read()
        if success == False:
            continue
        image2 = cv2.resize(image, (100, 50)) 
       # gray_image_2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        corr2_matrix = np.corrcoef(image1.reshape(-1), image2.reshape(-1))
        corr2 = corr2_matrix[0][1]
        corr_list.append(corr2)
        
    print video_name, np.median(corr_list)
    # if np.mean(corr_list) >= PARAM_CORR2D_COEFF:
    #     #move video in another folder
    #     shutil.move(video, PATH_problematic_videos + video_name)
