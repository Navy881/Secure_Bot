# -*- coding: utf-8 -*-

import os
import threading

import cv2
import imutils
import numpy as np
from datetime import datetime

from src.tools.video_record import save_image


class Camera(object):
    def __init__(self, cam=0, fps=30, record_video=False):
        self.camera_index = cam
        self.video = cv2.VideoCapture(self.camera_index)
        # self.video.set(cv2.CAP_PROP_FPS, fps)
        # self.video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        (self.grabbed, self.frame) = self.video.read()  # frame как атрибут объекта Camera

        self.updating = False
        self.update_frame_thread = None

        self.record_video = record_video
        self.video_filename = None
        self.video_out = None

    def __del__(self):
        self.stop()

    def start(self):
        self.updating = True
        if self.update_frame_thread is None or not self.update_frame_thread.is_alive():
            self.update_frame_thread = threading.Thread(target=self.update,
                                                        args=())  # получение видеопотка в отдельном потоке
            self.update_frame_thread.start()  # получение видеопотка в отдельном потоке
        else:
            print('WARNING: Camera is already starting')

        if self.record_video and self.video_out is None:
            self.create_video_file()

    def stop(self):
        self.updating = False
        if self.update_frame_thread is not None:
            self.update_frame_thread.join()
        if self.video_out is not None:
            self.video_out.release()
            self.video_out = None

    def get_frame(self):
        ret, frame = self.video.read()
        if ret:
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()

    # Обновление frame
    def update(self):
        while self.updating:
            (self.grabbed, self.frame) = self.video.read()

    def make_screenshot(self):
        ret, frame = self.video.read()
        if ret:
            cv2.imwrite('photo/bot_screenshot.png', frame)

    # Создание видеофайла
    def create_video_file(self):

        video_dir = "video/"

        date = datetime.now().strftime("%d-%m-%Y_(%H-%M-%S_%p)")
        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')  # Video format/Codec

        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        self.video_filename = video_dir + "camera" + str(self.camera_index) + "_" + date + ".avi"
        print("INFO: Create file: " + self.video_filename)

        self.video_out = cv2.VideoWriter(self.video_filename, fourcc, 30.0, (640, 480))

        if not self.video_out.isOpened():
            self.video_out = None

    def get_video_filename(self):
        return self.video_filename

    def get_video_out(self):
        return self.video_out

    # Детектирование движения
    def motion_detect(self, running, show_edges,
                      dnn_detection_status=False, net=None, classes=None, colors=None, given_confidence=0.2,
                      min_area=10, blur_size=11, blur_power=1, threshold_low=50, sending_period=60):
        first_frame = None

        fps = int(self.video.get(cv2.CAP_PROP_FPS))

        while running:

            text = "Unoccupied"

            if not self.grabbed:
                print("Sorry, camera is not found")
                break

            frame = self.frame.copy()

            frame1 = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (blur_size, blur_size), blur_power)

            # firstFrame = gray
            if first_frame is None:
                first_frame = gray
                continue

            frame_delta = cv2.absdiff(first_frame, gray)  # Difference between first and gray frames
            thresh = cv2.threshold(frame_delta, threshold_low, 255, cv2.THRESH_BINARY)[1]  # Frame_delta binarization
            thresh = cv2.dilate(thresh, None, iterations=2)  # Noise suppression

            cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cv2.putText(frame, "Camera 1", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, datetime.now().strftime("%d-%m-%Y %H:%M:%S%p"), (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            person_in_image = False

            for c in cnts:
                if cv2.contourArea(c) < min_area:
                    continue

                if show_edges:
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 0, 255), 2)

                text = "Occupied"
                # first_frame = gray

                # Запись кадра в файл
                if self.video_out is not None:
                    self.video_out.write(frame)

                if dnn_detection_status and net is not None and classes is not None and colors is not None:
                    # frame1 = self.real_time_detection(frame1, net, classes, colors, given_confidence)
                    frame1, persons = self.person_detection_on_image(frame1, net, classes, given_confidence)
                    if len(persons.keys()) > 0:
                        person_in_image = True

                if os.path.exists('photo/screenshot_temp.png'):
                    file_create_time = os.path.getmtime('photo/screenshot_temp.png')
                else:
                    file_create_time = 0

                send_delta = datetime.today().timestamp() - file_create_time
                if int(send_delta) > sending_period:
                    save_image(frame1)  # cv2.imwrite("photo/screenshot_temp.png", frame1)

                break

            cv2.putText(frame1, "Camera 1 {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame1, datetime.now().strftime("%d-%m-%Y %H:%M:%S%p"), (10, frame1.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(frame1, "FPS: " + str(fps), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            ret, jpeg = cv2.imencode('.jpg', frame1)  # For webapp

            return frame1, jpeg.tobytes(), text, person_in_image

    @staticmethod
    def real_time_detection(frame, net, classes, colors, given_confidence):
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > given_confidence:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = "{}: {:.2f}%".format(classes[idx], confidence * 100)
                print('INFO: detected {}'.format(label))

                # draw the prediction on the frame
                cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

        return frame

    def real_time_detection_2(self, dnn_detection_status, net, classes, colors, given_confidence):
        while dnn_detection_status:
            ret, frame = self.video.read()
            # frame = imutils.resize(frame, width=400)

            if not ret:
                print("Sorry, camera is not found")
                break

            detected_objects = dict()

            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

            # pass the blob through the network and obtain the detections and
            # predictions
            net.setInput(blob)
            detections = net.forward()

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > given_confidence:
                    # extract the index of the class label from the
                    # `detections`, then compute the (x, y)-coordinates of
                    # the bounding box for the object
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    label = "{}: {:.2f}%".format(classes[idx], confidence * 100)
                    print('INFO: detected {}'.format(label))

                    # Crop detected_objects
                    crop_detected_object = frame[startY:endY, startX:endX]
                    detected_objects[str(i) + '_' + label] = crop_detected_object

                    # draw the prediction on the frame
                    cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

            return frame, detected_objects

    def person_detection_on_video(self, dnn_detection_status, net, classes, given_confidence):
        while dnn_detection_status:
            ret, frame = self.video.read()
            # frame = imutils.resize(frame, width=400)

            if not ret:
                print("Sorry, camera is not found")
                break

            detected_objects = dict()

            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

            # pass the blob through the network and obtain the detections and
            # predictions
            net.setInput(blob)
            detections = net.forward()

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > given_confidence:
                    # extract the index of the class label from the
                    # `detections`, then compute the (x, y)-coordinates of
                    # the bounding box for the object
                    idx = int(detections[0, 0, i, 1])
                    if idx == 15:  # just person class 15
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        print('INFO: detected {}, confidence: {:.2f}%'.format(classes[idx], confidence * 100))

                        # Crop detected_objects
                        crop_detected_object = frame[startY:endY, startX:endX]
                        detected_objects[str(i) + '_' + classes[idx]] = crop_detected_object

                        # draw the prediction on the frame
                        line_length = int(min(endX - startX, endY - startY) / 6)
                        line_color = [0, 0, 255]
                        line_thickness = 1
                        text_color = [0, 0, 255]
                        text__thickness = 1
                        class_label = 'Class: {}'.format(classes[idx])
                        confidence_value = 'Confidence: {:.2f}%'.format(confidence * 100)

                        # Top left corner
                        cv2.line(img=frame,
                                 pt1=(startX, startY), pt2=(startX + line_length, startY),
                                 color=line_color,
                                 thickness=line_thickness)

                        cv2.line(img=frame,
                                 pt1=(startX, startY), pt2=(startX, startY + line_length),
                                 color=line_color,
                                 thickness=line_thickness)

                        # Top right corner
                        cv2.line(img=frame,
                                 pt1=(endX, startY), pt2=(endX - line_length, startY),
                                 color=line_color,
                                 thickness=line_thickness)

                        cv2.line(img=frame,
                                 pt1=(endX, startY), pt2=(endX, startY + line_length),
                                 color=line_color,
                                 thickness=line_thickness)

                        # Bottom left corner
                        cv2.line(img=frame,
                                 pt1=(startX, endY), pt2=(startX + line_length, endY),
                                 color=line_color,
                                 thickness=line_thickness)

                        cv2.line(img=frame,
                                 pt1=(startX, endY), pt2=(startX, endY - line_length),
                                 color=line_color,
                                 thickness=line_thickness)

                        # Bottom  right corner
                        cv2.line(img=frame,
                                 pt1=(endX, endY), pt2=(endX - line_length, endY),
                                 color=line_color,
                                 thickness=line_thickness)

                        cv2.line(img=frame,
                                 pt1=(endX, endY), pt2=(endX, endY - line_length),
                                 color=line_color,
                                 thickness=line_thickness)

                        # cv2.rectangle(img=frame,
                        #               pt1=(startX, startY), pt2=(endX, endY),
                        #               color=line_color,
                        #               thickness=line_thickness)

                        y = startY - 30 if startY - 30 > 30 else startY + 20
                        cv2.putText(img=frame,
                                    text=class_label,
                                    org=(startX + 5, y),
                                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                    fontScale=0.5,
                                    color=text_color,
                                    thickness=text__thickness)

                        cv2.putText(img=frame,
                                    text=confidence_value,
                                    org=(startX + 5, y + 20),
                                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                    fontScale=0.5,
                                    color=text_color,
                                    thickness=text__thickness)

            return frame, detected_objects

    @staticmethod
    def person_detection_on_image(frame, net, classes, given_confidence):
        detected_objects = dict()

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > given_confidence:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                if idx == 15:  # just person class 15
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    print('INFO: detected {}, confidence: {:.2f}%'.format(classes[idx], confidence * 100))

                    # Crop detected_objects
                    crop_detected_object = frame[startY:endY, startX:endX]
                    detected_objects[str(i) + '_' + classes[idx]] = crop_detected_object

                    # draw the prediction on the frame
                    line_length = int(min(endX - startX, endY - startY) / 6)
                    line_color = [0, 0, 255]
                    line_thickness = 1
                    text_color = [0, 0, 255]
                    text__thickness = 1
                    class_label = 'Class: {}'.format(classes[idx])
                    confidence_value = 'Confidence: {:.2f}%'.format(confidence * 100)

                    # Top left corner
                    cv2.line(img=frame,
                             pt1=(startX, startY), pt2=(startX + line_length, startY),
                             color=line_color,
                             thickness=line_thickness)

                    cv2.line(img=frame,
                             pt1=(startX, startY), pt2=(startX, startY + line_length),
                             color=line_color,
                             thickness=line_thickness)

                    # Top right corner
                    cv2.line(img=frame,
                             pt1=(endX, startY), pt2=(endX - line_length, startY),
                             color=line_color,
                             thickness=line_thickness)

                    cv2.line(img=frame,
                             pt1=(endX, startY), pt2=(endX, startY + line_length),
                             color=line_color,
                             thickness=line_thickness)

                    # Bottom left corner
                    cv2.line(img=frame,
                             pt1=(startX, endY), pt2=(startX + line_length, endY),
                             color=line_color,
                             thickness=line_thickness)

                    cv2.line(img=frame,
                             pt1=(startX, endY), pt2=(startX, endY - line_length),
                             color=line_color,
                             thickness=line_thickness)

                    # Bottom  right corner
                    cv2.line(img=frame,
                             pt1=(endX, endY), pt2=(endX - line_length, endY),
                             color=line_color,
                             thickness=line_thickness)

                    cv2.line(img=frame,
                             pt1=(endX, endY), pt2=(endX, endY - line_length),
                             color=line_color,
                             thickness=line_thickness)

                    # cv2.rectangle(img=frame,
                    #               pt1=(startX, startY), pt2=(endX, endY),
                    #               color=line_color,
                    #               thickness=line_thickness)

                    y = startY - 30 if startY - 30 > 30 else startY + 20
                    cv2.putText(img=frame,
                                text=class_label,
                                org=(startX + 5, y),
                                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                fontScale=0.5,
                                color=text_color,
                                thickness=text__thickness)

                    cv2.putText(img=frame,
                                text=confidence_value,
                                org=(startX + 5, y + 20),
                                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                fontScale=0.5,
                                color=text_color,
                                thickness=text__thickness)

        return frame, detected_objects

    def face_detection(self, dnn_detection_status, net, given_confidence):
        while dnn_detection_status:
            ret, frame = self.video.read()
            # frame = imutils.resize(frame, width=400)

            if not ret:
                print("Sorry, camera is not found")
                break

            detected_objects = list()

            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

            # pass the blob through the network and obtain the detections and
            # predictions
            net.setInput(blob)
            detections = net.forward()

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):

                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > given_confidence:
                    # extract the index of the class label from the
                    # `detections`, then compute the (x, y)-coordinates of
                    # the bounding box for the object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    label = "{:.2f}%".format(confidence * 100)
                    print('INFO: face detected {}'.format(label))

                    # Crop detected_objects
                    crop_detected_object = frame[startY:endY, startX:endX]
                    detected_objects.append(crop_detected_object)

                    # draw the prediction on the frame
                    line_length = int(min(endX - startX, endY - startY) / 6)
                    line_color = [0, 0, 255]
                    line_thickness = 1
                    text_color = [0, 0, 255]
                    text__thickness = 1
                    confidence_value = 'Face. Confidence: {:.2f}%'.format(confidence * 100)

                    # Top left corner
                    cv2.line(img=frame,
                             pt1=(startX, startY), pt2=(startX + line_length, startY),
                             color=line_color,
                             thickness=line_thickness)

                    cv2.line(img=frame,
                             pt1=(startX, startY), pt2=(startX, startY + line_length),
                             color=line_color,
                             thickness=line_thickness)

                    # Top right corner
                    cv2.line(img=frame,
                             pt1=(endX, startY), pt2=(endX - line_length, startY),
                             color=line_color,
                             thickness=line_thickness)

                    cv2.line(img=frame,
                             pt1=(endX, startY), pt2=(endX, startY + line_length),
                             color=line_color,
                             thickness=line_thickness)

                    # Bottom left corner
                    cv2.line(img=frame,
                             pt1=(startX, endY), pt2=(startX + line_length, endY),
                             color=line_color,
                             thickness=line_thickness)

                    cv2.line(img=frame,
                             pt1=(startX, endY), pt2=(startX, endY - line_length),
                             color=line_color,
                             thickness=line_thickness)

                    # Bottom  right corner
                    cv2.line(img=frame,
                             pt1=(endX, endY), pt2=(endX - line_length, endY),
                             color=line_color,
                             thickness=line_thickness)

                    cv2.line(img=frame,
                             pt1=(endX, endY), pt2=(endX, endY - line_length),
                             color=line_color,
                             thickness=line_thickness)

                    y = startY - 30 if startY - 30 > 30 else startY + 20
                    cv2.putText(img=frame,
                                text=confidence_value,
                                org=(startX + 5, y + 20),
                                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                fontScale=0.5,
                                color=text_color,
                                thickness=text__thickness)

            return frame, detected_objects

    def emotion_detection(self, dnn_detection_status, face_net, face_given_confidence,
                          emotion_net=None, emotion_classes=None, emotion_given_confidence=None):
        while dnn_detection_status:
            ret, frame = self.video.read()

            # frame = imutils.resize(frame, width=400)

            if not ret:
                print("Sorry, camera is not found")
                break

            emotion_detection = list()

            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

            # pass the blob through the network and obtain the detections and
            # predictions
            face_net.setInput(blob)
            detections = face_net.forward()

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):

                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > face_given_confidence:
                    # extract the index of the class label from the
                    # `detections`, then compute the (x, y)-coordinates of
                    # the bounding box for the object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    label = "{:.2f}%".format(confidence * 100)
                    # print('INFO: face detected {}'.format(label))

                    # draw the prediction on the frame
                    line_length = int(min(endX - startX, endY - startY) / 6)
                    line_color = [0, 0, 255]
                    line_thickness = 1
                    text_color = [0, 0, 255]
                    text__thickness = 1
                    confidence_value = 'Face. Confidence: {:.2f}%'.format(confidence * 100)

                    # Top left corner
                    cv2.line(img=frame,
                             pt1=(startX, startY), pt2=(startX + line_length, startY),
                             color=line_color,
                             thickness=line_thickness)

                    cv2.line(img=frame,
                             pt1=(startX, startY), pt2=(startX, startY + line_length),
                             color=line_color,
                             thickness=line_thickness)

                    # Top right corner
                    cv2.line(img=frame,
                             pt1=(endX, startY), pt2=(endX - line_length, startY),
                             color=line_color,
                             thickness=line_thickness)

                    cv2.line(img=frame,
                             pt1=(endX, startY), pt2=(endX, startY + line_length),
                             color=line_color,
                             thickness=line_thickness)

                    # Bottom left corner
                    cv2.line(img=frame,
                             pt1=(startX, endY), pt2=(startX + line_length, endY),
                             color=line_color,
                             thickness=line_thickness)

                    cv2.line(img=frame,
                             pt1=(startX, endY), pt2=(startX, endY - line_length),
                             color=line_color,
                             thickness=line_thickness)

                    # Bottom  right corner
                    cv2.line(img=frame,
                             pt1=(endX, endY), pt2=(endX - line_length, endY),
                             color=line_color,
                             thickness=line_thickness)

                    cv2.line(img=frame,
                             pt1=(endX, endY), pt2=(endX, endY - line_length),
                             color=line_color,
                             thickness=line_thickness)

                    y = startY - 40 if startY - 30 > 30 else startY + 20
                    cv2.putText(img=frame,
                                text=confidence_value,
                                org=(startX + 5, y),
                                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                fontScale=0.8,
                                color=text_color,
                                thickness=text__thickness)

                    # Emotion detection
                    # Crop detected_objects
                    crop_detected_object = frame[startY:endY, startX:endX]

                    emotion_detection = self.emotion_recognition(crop_detected_object, net=emotion_net)

                    emotion_confidence = max(emotion_detection[0])
                    idx = emotion_detection[0].tolist().index(emotion_confidence)
                    emotion_label = emotion_classes[idx]

                    if emotion_confidence > emotion_given_confidence:
                        emotion_value = 'Emotion: {} {:.2f}%'.format(emotion_label, emotion_confidence * 100)
                    else:
                        emotion_value = 'Emotion: ?'

                    cv2.putText(img=frame,
                                text=emotion_value,
                                org=(startX + 5, y + 30),
                                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                fontScale=0.8,
                                color=text_color,
                                thickness=text__thickness)

            return frame, emotion_detection

    @staticmethod
    def emotion_recognition(frame, net):
        # grab the frame dimensions and convert it to a blob
        # blob = cv2.dnn.blobFromImage(cv2.resize(frame, (224, 224)), 0.007843, (224, 224), )
        # blob = cv2.dnn.blobFromImage(cv2.resize(frame, (224, 224)), 1.0, (224, 224), (104.0, 177.0, 123.0))
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (224, 224)), 0.6, (224, 224), 170)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # confidence = max(detections[0])
        # idx = detections[0].tolist().index(confidence)
        # label = "{}: {:.2f}%".format(classes[idx], confidence * 100)
        # print('INFO: detected {}'.format(label))

        return detections
