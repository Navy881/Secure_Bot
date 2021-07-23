# -*- coding: utf-8 -*-

import json

# full_path = dirname(dirname(dirname(abspath(__file__))))  # /../../../
file_path = 'config/config.json'


class Config(object):

    conf = None

    def get_config(self):
        with open(file_path, 'r') as f:
            self.conf = json.loads(f.read())

    def _update(self):
        if self.conf is None:
            self.get_config()

    def __getitem__(self, key):
        self._update()
        return self.conf[key]

    def get(self, key, default=None):
        self._update()
        return self.conf.get(key, default)

    def invalidate(self):
        self.conf = None

    def update(self):
        with open(file_path, 'w') as f:
            json.dump(self.conf, f)
        self.get_config()


config = Config()


# # Open json-file for read
# def read_config():
#     # file_path = full_path+'/congig/config.json'
#     with open(file_path, 'r') as f:
#         data = json.loads(f.read())
#     return data
#
#
# # Write to json-file
# def edit_config(data):
#     # file_path = full_path+'/congig/config.json'
#     with open(file_path, 'w') as f:
#         json.dump(data, f)
#
#
# # Read detection parameters
# def get_detection_parameters():
#     data = read_config()
#
#     min_area = data['DetectionParameters']['min_area']
#     blur_size = data['DetectionParameters']['blur_size']
#     blur_power = data['DetectionParameters']['blur_power']
#     threshold_low = data['DetectionParameters']['threshold_low']
#
#     return min_area, blur_size, blur_power, threshold_low
#
#
# # Edit detection parameters
# def set_detection_parameters(min_area, blur_size, blur_power, threshold_low):
#     data = read_config()
#
#     data['DetectionParameters']['min_area'] = min_area
#     data['DetectionParameters']['blur_size'] = blur_size
#     data['DetectionParameters']['blur_power'] = blur_power
#     data['DetectionParameters']['threshold_low'] = threshold_low
#
#     edit_config(data)
#
#
# # Read bot parameters
# def get_bot_parameters():
#     data = read_config()
#
#     bot_token = data['BotParameters']['token']  # bot token
#     proxy_url = data['BotParameters']['proxy_url']  # proxy-server address
#     private_chat_id = data['BotParameters']['chat_id']  # chat id with bot
#     sending_period = data['BotParameters']['sending_period']  # message sending period in to chat by bot
#     username = data['BotParameters']['username']  # proxy username
#     password = data['BotParameters']['password']  # proxy password
#
#     # Proxy parameters
#     request_kwargs = {'proxy_url': proxy_url}
#
#     return bot_token, request_kwargs, private_chat_id, proxy_url, sending_period, username, password
#
#
# # Edit bot parameters
# def set_bot_parameters(bot_token, proxy_url, private_chat_id, sending_period, username, password):
#     data = read_config()
#
#     data['BotParameters']['token'] = bot_token  # bot token
#     data['BotParameters']['proxy_url'] = proxy_url  # proxy-server address
#     data['BotParameters']['chat_id'] = private_chat_id  # chat id with bot
#     data['BotParameters']['sending_period'] = sending_period  # message sending period in to chat by bot
#     data['BotParameters']['username'] = username  # proxy username
#     data['BotParameters']['password'] = password  # proxy password
#
#     edit_config(data)
#
#
# # Read NN parameters
# def get_nn_parameters():
#     data = read_config()
#
#     net_architecture = data['NNParameters']["face"]['architecture']  # net architecture
#     net_model = data['NNParameters']["face"]['model']  # net model
#     classes = data['NNParameters']["face"]['classes']  # classes
#     confidence = data['NNParameters']["face"]['confidence']  # confidence
#
#     return net_architecture, net_model, classes, confidence
