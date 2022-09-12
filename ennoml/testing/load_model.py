"""
 Author : Utkarsh
"""

import tensorflow as tf
from ennoml.models import Inception ,MobileNetV3, ShuffleNet, EffNet, MobileNet
from ennoml.util.extra_fn import logs
import sys


class load_model():
    
    def __init__(self, num_class, size, path, model_type, last_layer="mixed9"):
        self.model = None
        self.multi = False
        self.num_class = num_class
        self.size = size
        self.path = path
        self.model_type = model_type
        self.last_layer = last_layer
        if type(self.num_class) == list:
            self.multi = True

    def load(self):
        if not self.multi:
            if self.model_type in ['Inception', 'inception', 'InceptionV3', 'inceptionv3',"Multi_Type"]:
                try:
                    self.model = tf.keras.models.load_model(self.path)
                except:
                    model = Inception.V3(include_top=False,
                                         weights=None, input_shape=(self.size, self.size, 3))
                    layer = model.get_layer(self.last_layer)
                    layer = tf.keras.layers.Flatten()(layer.output)
                    output = tf.keras.layers.Dense(self.num_class, activation='softmax')(layer)
                    model = tf.keras.Model(model.input, output)
                    model.load_weights(self.path)
                return self.model
            elif self.model_type in ['MobileNet', 'mobilenet','MobileNetV2', 'mobilenetv2']:
                try:
                    self.model = tf.keras.models.load_model(self.path)
                except:
                    self.model = None
                    logs('Loading FAILED. Model Architecture did not load.',type='error',fn='print')
                    logs('Exiting the code.',type='error',fn='print')
                    sys.exit()
                return self.model
            elif self.model_type in ['MobileNetV3']:
                try:
                    self.model = tf.keras.models.load_model(self.path)
                except:
                    model = MobileNetV3.V3(type="small", input_shape=(self.size, self.size, 3), width_multiplier=1,
                                           divisible_by=8, l2_reg=2e-5, dropout_rate=0.2, name="MobileNetV3")
                    self.model = tf.keras.Sequential([
                        model,
                        tf.keras.layers.GlobalAveragePooling2D(),
                        tf.keras.layers.Dense(self.num_class, activation="softmax")])
                    self.model.load_weights(self.path)
                return self.model
            elif self.model_type in ['ShuffleNet']:
                try:
                    self.model = tf.keras.models.load_model(self.path)
                except:
                    self.model = ShuffleNet.V1((self.size, self.size, 3), self.num_class, include_top=True,
                                          weights=self.path)
                return self.model
            elif self.model_type in ['EffNet']:
                try:
                    self.model = tf.keras.models.load_model(self.path)
                except:
                    self.model = EffNet.V1((self.size, self.size, 3), self.num_class, include_top=True,
                                      weights=self.path)
                return self.model
            elif self.model_type == "Multi_Type":
                try:
                    self.model = tf.keras.models.load_model(self.path)
                except:
                    self.model = Inception.V3(include_top=False,
                                         weights=None, input_shape=(self.size, self.size, 3))
                    layer = self.model.get_layer(self.last_layer)
                    layer = tf.keras.layers.Flatten()(layer.output)
                    layer = tf.keras.layers.Dense(512)(layer)
                    output = tf.keras.layers.Dense(self.num_cls, activation=self.activation)(layer)
                    self.model = tf.keras.Model(self.model.input, output)
                    self.model.load_weights(self.path)
                return self.model
            else:
                try:
                    self.model = tf.keras.models.load_model(self.path)
                except:
                    self.model = None
                    logs('Loading FAILED. Model Architecture not found in the code.',type='error',fn='print')
                    logs('Exiting the code.',type='error',fn='print')
                    sys.exit()
                return self.model
        else:
            logs('WRONG MODEL TYPE. Model Architecture not found in the code.',type='error',fn='print')
            logs('Exiting the code.',type='error',fn='print')
            sys.exit()