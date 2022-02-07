#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import time
import logging
import requests
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup, KeyboardButton

from threading import Thread
from datetime import datetime
from re import findall
from subprocess import check_output

from src.camera import Camera
from src.tools.config import config


CAM = config["CameraParameters"]["camera_index"]
FPS = config["CameraParameters"]["fps"]
net_arch = config["NNParameters"]["object"]["architecture"]
net_model = config["NNParameters"]["object"]["model"]
net_confidence = config["NNParameters"]["object"]["confidence"]
classes = config["NNParameters"]["object"]["classes"]
min_area = config["DetectionParameters"]["min_area"]
blur_size = config["DetectionParameters"]["blur_size"]
blur_power = config["DetectionParameters"]["blur_power"]
threshold_low = config["DetectionParameters"]["threshold_low"]
bot_token = config["BotParameters"]["token"]
bot_private_chat_id = config["BotParameters"]["chat_id"]
bot_proxy_url = config["BotParameters"]["proxy_url"]
bot_sending_period = config["BotParameters"]["sending_period"]
bot_username = config["BotParameters"]["username"]
bot_password = config["BotParameters"]["password"]
open_exchange_rates_api_token = config["ServicesAPITokens"]["open_exchange_rates_token"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(net_arch, net_model)

telegram_bot = telebot.TeleBot(bot_token)


def grab():
    global detection_status, dnn_detection_status

    sending_time = 0

    while running:
        img, jpeg, detection_status, person_in_image = camera.motion_detect(running=running,
                                                                            show_edges=show_edges,
                                                                            dnn_detection_status=dnn_detection_status)
        if person_in_image:
            send_delta = datetime.today().timestamp() - sending_time
            if int(send_delta) > int(bot_sending_period):
                if NOTIFICATION_STATE:
                    send_image(bot_private_chat_id[0], jpeg)
                sending_time = datetime.now().timestamp()

        cv2.imshow("Camera", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def main_markup():
    markup = ReplyKeyboardMarkup()
    markup.row_width = 1
    markup.add("Options", "Help")
    return markup


def admin_menu_markup():
    markup = ReplyKeyboardMarkup()
    markup.row_width = 3
    markup.add(CAMERA_OPTION, "Get photo", "Get last video",
               NOTIFICATION_OPTION, "Get temperature", "Check health",
               "Get exchange rates", "Return")
    return markup


def user_menu_markup():
    markup = ReplyKeyboardMarkup()
    markup.row_width = 3
    markup.add("Get exchange rates", "Return")
    return markup


def delete_message_markup():
    markup = InlineKeyboardMarkup()
    markup.row_width = 1
    markup.add(InlineKeyboardButton("Delete", callback_data="cb_delete"))
    return markup


def showing_options(chat_type, chat_id):
    now = datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S")
    if str(chat_id) in bot_private_chat_id:
        telegram_bot.send_message(chat_id=chat_id, text='Admin bot options', reply_markup=admin_menu_markup())
    else:
        telegram_bot.send_message(chat_id=chat_id, text='Bot options', reply_markup=user_menu_markup())
    print("{} - bot answered into {} chat: {}".format(now, chat_type, chat_id))


def switch_camera(chat_type, chat_id):
    global camera, running, capture_thread, CAMERA_OPTION

    if str(chat_id) in bot_private_chat_id:
        if not running:
            print("Camera starting")
            camera.start()
            running = True
            CAMERA_OPTION = 'Stop camera'
            capture_thread = Thread(target=grab, args=())  # Thread for detector
            capture_thread.start()
            print("Camera started")
            message_text = "Record started..."
        else:
            print("Camera stopping")
            running = False
            CAMERA_OPTION = 'Start camera'
            capture_thread.join()
            camera.stop()
            print("Camera stopped")
            message_text = "Record stopped..."

        now = datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S")
        telegram_bot.send_message(chat_id=chat_id, text=message_text, reply_markup=admin_menu_markup())
        print("{} - bot answered into {} chat: {}".format(now, chat_type, chat_id))


def send_help_info(chat_type, chat_id):
    now = datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S")
    if str(chat_id) in bot_private_chat_id:
        help_message_text = "You can use commands:\n" \
                           " - press <Check health> to check health\n" \
                           " - press <Start camera> to start camera\n" \
                           " - press <Stop camera> to stop camera\n" \
                           " - press <Get photo> to get photo\n" \
                           " - press <Get temperature> to get temperature\n" \
                           " - press <Help> to get help info\n"
    else:
        help_message_text = "You can use commands:\n" \

    telegram_bot.send_message(chat_id=chat_id, text=help_message_text, reply_markup=main_markup())
    print("{} - bot answered into {} chat: {}".format(now, chat_type, chat_id))


def send_sign_of_life(chat_type, chat_id):
    now = datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S")
    telegram_bot.send_message(chat_id=chat_id, text="I'm work")
    print("{} - bot answered into {} chat: {}".format(now, chat_type, chat_id))


def send_photo(chat_type, chat_id):
    if str(chat_id) in bot_private_chat_id and camera is not None:
        camera.make_screenshot()  # Make screenshot
        now = datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S")
        time.sleep(1)
        telegram_bot.send_photo(chat_id=chat_id,
                                photo=open('photo/bot_screenshot.png', 'rb'),
                                reply_markup=delete_message_markup())
        print("{} - bot sent image into {} chat: {}".format(now, chat_type, chat_id))


def send_image(chat_id, image):
    now = datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S")
    telegram_bot.send_photo(chat_id=chat_id, photo=image, reply_markup=delete_message_markup())
    print("{} - bot sent image into private chat: {}".format(now, chat_id))


def send_temp(chat_type, chat_id):
    if str(chat_id) in bot_private_chat_id:
        temp = check_output(["vcgencmd", "measure_temp"]).decode()  # Request temperature
        temp = float(findall('\d+\.\d+', temp)[0])
        message_text = "Temperature is " + str(temp) + " C"
        now = datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S")
        telegram_bot.send_message(chat_id=chat_id, text=message_text)
        print("{} - bot answered into {} chat: {}".format(now, chat_type, chat_id))


def send_last_video(chat_type, chat_id):
    path = 'video/'
    last_file_created_on = 0
    last_file = None

    if str(chat_id) in bot_private_chat_id:
        if os.path.exists(path):
            for filename in os.listdir(path):
                file_created_on = os.path.getmtime(path + filename)
                if file_created_on > last_file_created_on:
                    last_file = path + filename
                last_file_created_on = file_created_on

            if last_file is not None:
                now = datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S")
                telegram_bot.send_video(chat_id=chat_id,
                                        video=open(last_file, 'rb'),
                                        reply_markup=delete_message_markup())
                print("{} - bot sent video into {} chat: {}".format(now, chat_type, chat_id))
            else:
                message_text = "Video not found"
                now = datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S")
                telegram_bot.send_message(chat_id=chat_id, text=message_text)
                print("{} - bot answered into {} chat: {}".format(now, chat_type, chat_id))


def switch_notifications(chat_type, chat_id):
    global NOTIFICATION_STATE, NOTIFICATION_OPTION

    if str(chat_id) in bot_private_chat_id:
        if NOTIFICATION_STATE:
            NOTIFICATION_STATE = False
            NOTIFICATION_OPTION = 'Enable notifications'
            message_text = "Notifications disabled..."
        else:
            NOTIFICATION_STATE = True
            NOTIFICATION_OPTION = 'Disable notifications'
            message_text = "Notifications enabled..."

        now = datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S")
        telegram_bot.send_message(chat_id=chat_id, text=message_text, reply_markup=admin_menu_markup())
        print("{} - bot answered into {} chat: {}".format(now, chat_type, chat_id))


def return_to_main_menu(chat_type, chat_id):
    now = datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S")
    telegram_bot.send_message(chat_id=chat_id, text='Return', reply_markup=main_markup())
    print("{} - bot answered into {} chat: {}".format(now, chat_type, chat_id))


def send_exchange_rates(chat_type, chat_id):
    currencies = ["GBP", "USD", "EUR", "CNY", "JPY"]
    r = requests.get(url="https://www.cbr-xml-daily.ru/latest.js?app_id=" + open_exchange_rates_api_token + "&base=RUB")
    if r.status_code == 200:
        result = r.json()
        if "rates" in result:
            rates = result["rates"]
            message_text = "Exchange rates:\n"
            for rate in currencies:
                if rate in rates:
                    message_text += "• {}:  {:.4f}\n".format(rate, 1/rates[rate])
            now = datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S")
            telegram_bot.send_message(chat_id=chat_id, text=message_text)
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
                telegram_bot.send_photo(chat_id=bot_private_chat_id, photo=open('photo/screenshot_temp.png', 'rb'))
                print("{} - bot sent image into chat: {}".format(now, bot_private_chat_id))
                send_time = os.path.getmtime('photo/screenshot_temp.png')


"""
Commands for bot
"""
commands = {
    "Admin": {
        "options": showing_options,
        "start camera": switch_camera,
        "stop camera": switch_camera,
        "get photo": send_photo,
        "get last video": send_last_video,
        "disable notifications": switch_notifications,
        "enable notifications": switch_notifications,
        "get temperature": send_temp,
        "check health": send_sign_of_life,
        "help": send_help_info,
        "return": return_to_main_menu,
        "get exchange rates": send_exchange_rates
    },
    "User": {
        "options": showing_options,
        "help": send_help_info,
        "return": return_to_main_menu,
        "get exchange rates": send_exchange_rates
    }
}


@telegram_bot.message_handler(func=lambda message: True and message.content_type == 'text')
def main(message):
    chat_id = message.chat.id
    chat_type = message.chat.type
    message_text = message.text.lower()
    if str(chat_id) in bot_private_chat_id:
        if message_text in commands['Admin'].keys():
            commands['Admin'].get(message_text)(chat_type, chat_id)
        else:
            send_help_info(chat_type, chat_id)
    else:
        if message_text in commands['User'].keys():
            commands['User'].get(message_text)(chat_type, chat_id)
        else:
            send_help_info(chat_type, chat_id)


@telegram_bot.callback_query_handler(func=lambda call: True)
def callback_query(call):
    if call.data == "cb_delete":
        telegram_bot.answer_callback_query(call.id, "Deleting the message...")
        telegram_bot.delete_message(chat_id=call.message.chat.id, message_id=call.message.message_id)


if __name__ == '__main__':

    running = False
    show_edges = False
    alarm_bot_status = False
    dnn_detection_status = True
    capture_thread = None
    detection_status = ''
    CAMERA_OPTION = 'Start camera'
    NOTIFICATION_STATE = True
    NOTIFICATION_OPTION = 'Disable notifications'

    star_time = datetime.now().replace(microsecond=0)
    send_time = datetime.today().timestamp()

    camera = Camera(camera_idx=CAM,
                    fps=FPS,
                    record_video=True,
                    min_area=int(min_area),
                    blur_size=int(blur_size),
                    blur_power=int(blur_power),
                    threshold_low=int(threshold_low),
                    net=net,
                    detection_classes=classes,
                    confidence=float(net_confidence),
                    classes_colors=np.random.uniform(0, 255, size=(len(classes), 3)))

    while True:
        try:
            print("bot running...")
            logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
            logger = logging.getLogger(__name__)
            telegram_bot.polling(none_stop=True)
            # telegram_bot.infinity_polling()  # polling(none_stop=True, timeout=50)
            # alarm()
            # while 1:
            #     time.sleep(10)
        except Exception as e:
            logging.error(e)
            time.sleep(5)
            print("Internet error!")
