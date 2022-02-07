# -*- coding: utf-8 -*-

import os
import threading

import cv2
import imutils
import numpy as np
from datetime import datetime, timedelta

from src.tools.video_record import save_image


class DetectedObject(object):
    def __init__(self, coordinates=(0, 0, 0, 0), label=None, confidence=0.0, image=None):
        self.coordinates = coordinates
        self.label = label
        self.confidence = confidence
        self.image = image


# noinspection PyUnresolvedReferences
class Camera(object):
    def __init__(self, camera_idx=0, fps=30, record_video=False,
                 min_area=10, blur_size=11, blur_power=1, threshold_low=50, sending_period=60,
                 net=None, detection_classes=list, confidence=0.6, classes_colors=None, detection_period_in_seconds=1):
        self.camera_index = camera_idx
        self.video = cv2.VideoCapture(self.camera_index)
        # self.video.set(cv2.CAP_PROP_FPS, fps)
        # self.video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        '''
        frame как атрибует объекта Camera
        Сокращает время на первом запуске
        Обновляется в отдельном потоке при вызове start()
        '''
        (self.grabbed, self.frame) = self.video.read()

        self.updating = False
        self.update_frame_thread = None

        self.record_video = record_video
        self.video_filename = None
        self.video_out = None

        self.no_signal_image = np.zeros((600, 600, 3), np.uint8)
        cv2.putText(img=self.no_signal_image, text="Camera is not found", org=(120, 300),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=[0, 0, 255], thickness=1)

        # Parameters for motion detection
        self.min_area = min_area
        self.blur_size = blur_size
        self.blur_power = blur_power
        self.threshold_low = threshold_low
        self.sending_period = sending_period

        # Parameters for detection
        self.net = net
        self.detection_classes = detection_classes
        self.given_confidence = confidence
        self.classes_colors = classes_colors
        self.detection_period_in_seconds = detection_period_in_seconds

        self.last_detection_time = datetime.now()
        self.detected_objects = []

    def __del__(self):
        self.stop()

    def start(self):
        self.updating = True
        if self.update_frame_thread is None or not self.update_frame_thread.is_alive():
            self.update_frame_thread = threading.Thread(target=self.update,
                                                        args=())
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
            if not self.grabbed:
                print("Sorry, camera is not found")
                self.video = cv2.VideoCapture(self.camera_index)
                self.frame = self.no_signal_image
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

    # draw the prediction on the frame
    def draw_bounds(self, frame=None):

        if frame is None:
            frame = self.frame

        for obj in self.detected_objects:
            (startX, startY, endX, endY) = obj.coordinates
            confidence = obj.confidence
            class_label = obj.label

            line_length = int(min(endX - startX, endY - startY) / 6)
            line_color = [0, 0, 255]
            line_thickness = 1

            text_color = [0, 0, 255]
            text__thickness = 1

            class_label = 'Class: {}'.format(class_label)

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

    # updated 26.12.2021
    # Детектирование движения
    def motion_detect(self, running=True, show_edges=True, dnn_detection_status=False):
        first_frame = None

        while running:

            text = "Unoccupied"

            if not self.grabbed:
                _, jpeg = cv2.imencode('.jpg', self.no_signal_image)  # For webapp
                return self.no_signal_image, jpeg.tobytes(), text, False

            fps = int(self.video.get(cv2.CAP_PROP_FPS))

            frame = self.frame.copy()

            frame_resized = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), self.blur_power)

            # firstFrame = gray
            if first_frame is None:
                first_frame = gray
                continue

            frame_delta = cv2.absdiff(first_frame, gray)  # Difference between first and gray frames
            thresh = cv2.threshold(frame_delta, self.threshold_low, 255, cv2.THRESH_BINARY)[1]  # Frame_delta binarization
            thresh = cv2.dilate(thresh, None, iterations=2)  # Noise suppression

            cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cv2.putText(frame, "Camera 1", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, datetime.now().strftime("%d-%m-%Y %H:%M:%S%p"), (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            person_in_image = False

            for c in cnts:
                if cv2.contourArea(c) < self.min_area:
                    continue

                if show_edges:
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 0, 255), 2)

                text = "Occupied"
                # first_frame = gray

                # Запись кадра в файл
                if self.video_out is not None:
                    self.video_out.write(frame)

                if dnn_detection_status:
                    # frame1 = self.real_time_detection(frame1, net, classes, colors, given_confidence)
                    frame_resized = self.person_detection_on_image(frame_resized)
                    if len(self.detected_objects) > 0:
                        person_in_image = True

                if os.path.exists('photo/screenshot_temp.png'):
                    file_create_time = os.path.getmtime('photo/screenshot_temp.png')
                else:
                    file_create_time = 0

                send_delta = datetime.today().timestamp() - file_create_time
                if int(send_delta) > self.sending_period:
                    save_image(frame_resized)  # cv2.imwrite("photo/screenshot_temp.png", frame1)

                break

            cv2.putText(frame_resized, "Camera 1 {}".format(text), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame_resized, datetime.now().strftime("%d-%m-%Y %H:%M:%S%p"),
                        (10, frame_resized.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(frame_resized, "FPS: " + str(fps), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            ret, jpeg = cv2.imencode('.jpg', frame_resized)  # For webapp

            return frame_resized, jpeg.tobytes(), text, person_in_image

    # updated 26.12.2021
    def person_detection_on_image(self, frame):
        if self.net is None or self.detection_classes is None:
            return frame

        self.detected_objects = []

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        self.net.setInput(blob)
        detections = self.net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > self.given_confidence:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                if idx == 15:  # just person class 15
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Crop detected_objects
                    crop_detected_object = self.frame[startY:endY, startX:endX]

                    # Add detected objects into self list
                    self.detected_objects.append(DetectedObject((startX, startY, endX, endY),
                                                                self.detection_classes[idx],
                                                                confidence,
                                                                crop_detected_object))

                    print('INFO: detected {}, confidence: {:.2f}%'.format(self.detection_classes[idx],
                                                                          confidence * 100))

                    # draw the prediction on the frame
                    self.draw_bounds(frame)

        return frame

    # updated 26.12.2021
    def person_detection_on_video(self, dnn_detection_status=True):

        while dnn_detection_status:
            if not self.grabbed:
                return self.no_signal_image

            # frame = imutils.resize(frame, width=400)

            # Detect objects every few second
            if datetime.now() - self.last_detection_time >= timedelta(seconds=self.detection_period_in_seconds):
                self.last_detection_time = datetime.now()

                self.detected_objects = []

                # grab the frame dimensions and convert it to a blob
                (h, w) = self.frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(self.frame, (300, 300)), 0.007843, (300, 300), 127.5)

                # pass the blob through the network and obtain the detections and
                # predictions
                self.net.setInput(blob)
                detections = self.net.forward()

                # loop over the detections
                for i in np.arange(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated with
                    # the prediction
                    confidence = detections[0, 0, i, 2]

                    # filter out weak detections by ensuring the `confidence` is
                    # greater than the minimum confidence
                    if confidence > self.given_confidence:
                        # extract the index of the class label from the
                        # `detections`, then compute the (x, y)-coordinates of
                        # the bounding box for the object
                        idx = int(detections[0, 0, i, 1])
                        if idx == 15:  # just person class 15
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")

                            # Crop detected_objects
                            crop_detected_object = self.frame[startY:endY, startX:endX]

                            # Add detected objects into self list
                            self.detected_objects.append(DetectedObject((startX, startY, endX, endY),
                                                                        self.detection_classes[idx],
                                                                        confidence,
                                                                        crop_detected_object))

                            print('INFO: detected {}, confidence: {:.2f}%'.format(self.detection_classes[idx],
                                                                                  confidence * 100))

            self.draw_bounds()

            return self.frame

    # updated 26.12.2021
    def emotion_detection(self, dnn_detection_status=True, emotion_net=None):
        while dnn_detection_status:

            if not self.grabbed:
                return self.no_signal_image

            # Detect objects every few second
            if datetime.now() - self.last_detection_time >= timedelta(seconds=self.detection_period_in_seconds):
                self.last_detection_time = datetime.now()

                self.detected_objects = []

                # grab the frame dimensions and convert it to a blob
                (h, w) = self.frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(self.frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

                # pass the blob through the network and obtain the detections and
                # predictions
                self.net.setInput(blob)
                detections = self.net.forward()

                # loop over the detections
                for i in np.arange(0, detections.shape[2]):

                    # extract the confidence (i.e., probability) associated with
                    # the prediction
                    confidence = detections[0, 0, i, 2]

                    # filter out weak detections by ensuring the `confidence` is
                    # greater than the minimum confidence
                    if confidence > self.given_confidence:
                        # extract the index of the class label from the
                        # `detections`, then compute the (x, y)-coordinates of
                        # the bounding box for the object
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        crop_detected_object = self.frame[startY:endY, startX:endX]

                        emotion_label, emotion_confidence = self.emotion_recognition(crop_detected_object,
                                                                                     net=emotion_net)

                        # Add detected objects into self list
                        self.detected_objects.append(DetectedObject((startX, startY, endX, endY),
                                                                    emotion_label,
                                                                    emotion_confidence,
                                                                    crop_detected_object))

            self.draw_bounds()

            return self.frame

    # updated 26.12.2021
    def emotion_recognition(self, frame, net):
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

        emotion_confidence = max(detections[0])
        idx = detections[0].tolist().index(emotion_confidence)
        emotion_label = self.detection_classes[idx]

        return emotion_label, emotion_confidence

    '''
    Not updated
    '''
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
