"""
 Author : Utkarsh
"""

from ennoml.models import Inception, MobileNet, MobileNetV3, ShuffleNet, EffNet
import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from ennoml.util.extra_fn import logs

class single():

    def __init__(self, num_cls, size, weight_file, retrain,
                 loss_fn=tf.keras.losses.categorical_crossentropy,
                 opti=tf.keras.optimizers.SGD(learning_rate=0.001),
                 metric=['categorical_accuracy'],
                 activation='softmax',
                 last_layer='mixed9'):
        self.num_cls = num_cls
        self.size = size
        self.weight_file = weight_file
        self.loss_fn = loss_fn
        self.opti = opti
        self.metric = metric
        self.activation = activation
        self.last_layer = last_layer
        self.retrain = retrain


    def InceptionV3(self):
        if self.retrain:
            model = tf.keras.models.load_model(self.weight_file)
            for layer in model.layers:
                layer.trainable = True
            last_layer = model.layers[-3].name
            lastlayer = model.get_layer(last_layer)
            lastlayer = tf.keras.layers.Flatten()(lastlayer.output)
            Output = tf.keras.layers.Dense(self.num_cls, activation=self.activation)(lastlayer)
            model = tf.keras.Model(model.input, Output)
            model.compile(optimizer=self.opti, loss=self.loss_fn, metrics=self.metric)
            logs(f"Model created and compiled for, {self.num_cls} Class",type='logs',fn='print')
            logs(f"Loaded Pretrained Weights - {self.weight_file}",type='logs',fn='print')
            return model

        else:
            model = Inception.V3(input_shape=(self.size, self.size, 3), include_top=False, weights=None)
            try:
                model.load_weights(self.weight_file)
            except:
                logs(f"No Weight File Found",type='logs',fn='print')
                logs(f"Building MODEL Without Weight",type='logs',fn='print')
            if self.last_layer == "Auto":
                self.last_layer = model.layers[-1].name
                layer = model.get_layer(self.last_layer)
                layer = tf.keras.layers.Flatten()(layer.output)
            else:
                try:
                    layer = model.get_layer(self.last_layer)
                    layer = tf.keras.layers.Flatten()(layer.output)
                except:
                    self.last_layer = model.layers[-1].name
                    layer = model.get_layer(self.last_layer)
                    layer = tf.keras.layers.Flatten()(layer.output)
            output = tf.keras.layers.Dense(self.num_cls, activation=self.activation)(layer)
            model = tf.keras.Model(model.input, output)
            model.compile(optimizer=self.opti, loss=self.loss_fn, metrics=self.metric)
            logs(f"Model created and compiled for, {self.num_cls} Class",type='logs',fn='print')
            return model

    def MobileNetV2(self,alpha=1):
        if self.retrain:
            model = tf.keras.models.load_model(self.weight_file)
            last_layer = model.layers[-3].name
            lastlayer = model.get_layer(last_layer)
            model = tf.keras.Sequential([
                lastlayer,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(self.num_cls, activation=self.activation)])
            model.compile(optimizer=self.opti, loss=self.loss_fn, metrics=self.metric)
            logs(f"Model created and compiled for, {self.num_cls} Class",type='logs',fn='print')
            logs(f"Loaded Pretrained Weights - {self.weight_file}",type='logs',fn='print')
            return model
        else:
            model = MobileNet.V2(input_shape=(self.size, self.size, 3), include_top=False, weights=None,alpha=alpha)
            for layer in model.layers:
                layer.trainable = True
            model = tf.keras.Sequential([
                model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(self.num_cls, activation=self.activation)])
            model.compile(optimizer=self.opti, loss=self.loss_fn, metrics=self.metric)
            logs(f"Model created and compiled for, {self.num_cls} Class",type='logs',fn='print')
            return model

    def MobileNetV3(self,alpha=1):
        if self.retrain:
            model = tf.keras.models.load_model(self.weight_file)
            last_layer = model.layers[-3].name
            lastlayer = model.get_layer(last_layer)
            model = tf.keras.Sequential([
                lastlayer,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(self.num_cls, activation=self.activation)])
            model.compile(optimizer=self.opti, loss=self.loss_fn, metrics=self.metric)
            logs(f"Model created and compiled for, {self.num_cls} Class",type='logs',fn='print')
            logs(f"Loaded Pretrained Weights - {self.weight_file}",type='logs',fn='print')
            return model
        else:
            model = MobileNetV3.V3(type="small", input_shape=(self.size, self.size, 3), width_multiplier=alpha,
                divisible_by=8, l2_reg=2e-5, dropout_rate=0.2, name="MobileNetV3small")
            model = tf.keras.Sequential([
                model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(self.num_cls, activation=self.activation)])
            model.compile(optimizer=self.opti, loss=self.loss_fn, metrics=self.metric)
            logs(f"Model created and compiled for, {self.num_cls} Class",type='logs',fn='print')
            return model

    def ShuffleNet(self):
        if self.retrain:
            try:
                model = tf.keras.models.load_model(self.weight_file)
            except:
                model = ShuffleNet.V1((self.size,self.size,3),self.num_cls,include_top=False,weights=self.weight_file)
            last_layer = model.layers[-3].name
            lastlayer = model.get_layer(last_layer)
            lastlayer = tf.keras.layers.GlobalAveragePooling2D()(lastlayer.output)
            Output = tf.keras.layers.Dense(self.num_cls, activation=self.activation)(lastlayer)
            model = tf.keras.Model(model.input, Output)
            model.compile(optimizer=self.opti, loss=self.loss_fn, metrics=self.metric)
            logs(f"Model created and compiled for, {self.num_cls} Class",type='logs',fn='print')
            logs(f"Loaded Pretrained Weights - {self.weight_file}",type='logs',fn='print')
            return model
        else:
            model = ShuffleNet.V1((self.size,self.size,3),self.num_cls,include_top=True,weights=None)
            model.compile(optimizer=self.opti, loss=self.loss_fn, metrics=self.metric)
            logs(f"Model created and compiled for, {self.num_cls} Class",type='logs',fn='print')
            return model

    def EffNet(self):
        if self.retrain:
            try:
                model = tf.keras.models.load_model(self.weight_file)
            except:
                model = EffNet.V1((self.size,self.size,3),self.num_cls,include_top=False,weights=self.weight_file)
            last_layer = model.layers[-3].name
            lastlayer = model.get_layer(last_layer)
            lastlayer = tf.keras.layers.Flatten()(lastlayer.output)
            Output = tf.keras.layers.Dense(self.num_cls, activation=self.activation)(lastlayer)
            model = tf.keras.Model(model.input, Output)
            model.compile(optimizer=self.opti, loss=self.loss_fn, metrics=self.metric)
            logs(f"Model created and compiled for, {self.num_cls} Class",type='logs',fn='print')
            logs(f"Loaded Pretrained Weights - {self.weight_file}",type='logs',fn='print')
            return model
        else:
            model = EffNet.V1((self.size,self.size,3),self.num_cls,include_top=True,weights=None)
            model.compile(optimizer=self.opti, loss=self.loss_fn, metrics=self.metric)
            logs(f"Model created and compiled for, {self.num_cls} Class",type='logs',fn='print')
            return model
