# -*- coding: utf-8 -*-

import os
import cv2
from datetime import datetime

video_dir = "video/"
photo_dir = "photo/"


def create_video():
    date = datetime.now().strftime("%d-%m-%Y_(%H-%M-%S_%p)")
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')  # Video format

    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    videofile_name = video_dir+date+".avi"
    print(videofile_name)
    out = cv2.VideoWriter(videofile_name, fourcc, 30.0, (640, 480))
    return out, videofile_name


def record(out, frame):
    out.write(frame)


def save_image(frame):
    if not os.path.exists(photo_dir):
        os.makedirs(photo_dir)
    cv2.imwrite(photo_dir+"screenshot_temp.png", frame)
