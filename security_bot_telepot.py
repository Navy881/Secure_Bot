#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import time
import logging
import telepot
from threading import Thread
from datetime import datetime
from telepot.loop import MessageLoop
from re import findall
from subprocess import check_output

from src.camera import Camera
from src.tools.param_manage import get_detection_parameters, get_bot_parameters, get_nn_parameters
from src.tools.video_record import create_video

running = False
show_edges = True
alarm_bot_status = False
dnn_detection_status = False
capture_thread = None
detection_status = ''
v_filename = ''

star_time = datetime.now().replace(microsecond=0)

min_area, blur_size, blur_power, threshold_low = get_detection_parameters()
bot_token, request_kwargs, private_chat_id, proxy_url, sending_period, username, password = get_bot_parameters()
net_architecture, net_model, classes, confidence = get_nn_parameters()

SetProxy = telepot.api.set_proxy(proxy_url, (username, password))
bot = telepot.Bot(bot_token)

CAM = 0  # TODO to config
FPS = 10  # TODO to config
camera = Camera(CAM, FPS)

send_time = datetime.today().timestamp()


def grab():
    global v_filename, detection_status, dnn_detection_status

    out, v_filename = create_video()
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(net_architecture, net_model)  # Load serialized model from disk
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    while running:
        img, detection_status = camera.motion_detect(running, out, show_edges, dnn_detection_status, net,
                                                     classes, colors, float(confidence), int(min_area),
                                                     int(blur_size), int(blur_size), int(threshold_low),
                                                     int(sending_period))
        cv2.imshow("Camera", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def start_camera(chat_type, chat_id):
    global running, capture_thread

    if running is False:
        print("Starting")
        running = True
        capture_thread = Thread(target=grab, args=())  # Thread for detector
        capture_thread.start()
        print("Started")
        message = "Record started..."
    else:
        message = "Record already started..."

    now = datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S")
    bot.sendMessage(chat_id, message)
    print("{} - bot answered into {} chat: {}".format(now, chat_type, chat_id))


def stop_camera(chat_type, chat_id):
    global running, capture_thread

    if running is True:
        print("Stopping")
        running = False
        capture_thread.join()
        print("Stopped")
        message = "Record stopped..."
    else:
        message = "Nothing running..."

    now = datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S")
    bot.sendMessage(chat_id, message)
    print("{} - bot answered into {} chat: {}".format(now, chat_type, chat_id))


def send_help_info(chat_type, chat_id):
    now = datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S")
    help_message = "You can use commands:\n" \
                   "/check - checking work,\n" \
                   "/help - help info\n" \
                   "/photo - sending photo in chat\n" \
                   "/start_camera - starting detection and recording\n" \
                   "/stop_camera - stopping detection and recording\n" \
                   "/temp - get temperature value\n"

    bot.sendMessage(chat_id, help_message)
    print("{} - bot answered into {} chat: {}".format(now, chat_type, chat_id))


def send_sign_of_life(chat_type, chat_id):
    now = datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S")
    bot.sendMessage(chat_id, text="I'm work")
    print("{} - bot answered into {} chat: {}".format(now, chat_type, chat_id))


def send_photo(chat_type, chat_id):
    camera.make_screenshot()  # Make screenshot
    now = datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S")
    bot.sendPhoto(chat_id, photo=open('photo/bot_screenshot.png', 'rb'))
    print("{} - bot sent image into {} chat: {}".format(now, chat_type, chat_id))


def get_temp(chat_type, chat_id):
    temp = check_output(["vcgencmd", "measure_temp"]).decode()  # Request temperature
    temp = float(findall('\d+\.\d+', temp)[0])
    message = "Temperature is " + str(temp) + " C"
    now = datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S")
    bot.sendMessage(chat_id, message)
    print("{} - bot answered into {} chat: {}".format(now, chat_type, chat_id))


def alarm():
    global send_time
    while alarm_bot_status:
        if os.path.exists('photo/screenshot_temp.png'):
            file_create_time = os.path.getmtime('photo/screenshot_temp.png')
            if send_time != file_create_time:
                time.sleep(1)
                print("sending image...")
                now = datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S")
                bot.sendPhoto(private_chat_id, photo=open('photo/screenshot_temp.png', 'rb'))
                print("{} - bot sent image into chat: {}".format(now, private_chat_id))
                send_time = os.path.getmtime('photo/screenshot_temp.png')


"""
Commands for bot
"""
commands = {
    "/check": send_sign_of_life,
    "/help": send_help_info,
    "/photo": send_photo,
    "/start_camera": start_camera,
    "/stop_camera": stop_camera,
    "/temp": get_temp,
}


def main(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    if content_type == 'text':
        command = msg['text']
        if command in commands.keys():
            commands.get(command)(chat_type, chat_id)
        else:
            send_help_info(chat_type, chat_id)


if __name__ == '__main__':
    try:
        print("bot running...")
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
        logger = logging.getLogger(__name__)
        MessageLoop(bot, main).run_as_thread()
        alarm()
        while 1:
            time.sleep(10)
    except Exception as e:
        print(e)
