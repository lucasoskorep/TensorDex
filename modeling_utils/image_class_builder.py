import random
from enum import Enum
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .model_wrapper import ModelWrapper


class ImageClassModels(Enum):
    INCEPTION_V3 = ModelWrapper(
        keras.applications.inception_v3.InceptionV3,
        keras.applications.inception_v3.preprocess_input,
        "inception_v3"
    )
    XCEPTION = ModelWrapper(
        keras.applications.xception.Xception,
        keras.applications.xception.preprocess_input,
        "xception"
    )
    MOBILENET_V2 = ModelWrapper(
        keras.applications.mobilenet_v2.MobileNetV2,
        keras.applications.mobilenet_v2.preprocess_input,
        "mobilenet_v2"
    )
    EFFICIENTNET_V2S = ModelWrapper(
        keras.applications.efficientnet_v2.EfficientNetV2S,
        tf.keras.applications.efficientnet_v2.preprocess_input,
        "efficientnet_v2s"
    )
    EFFICIENTNET_V2B0 = ModelWrapper(
        keras.applications.efficientnet_v2.EfficientNetV2B0,
        tf.keras.applications.efficientnet_v2.preprocess_input,
        "efficientnet_v2b0"

    )


class ImageClassModelBuilder(object):

    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 n_classes: int,
                 optimizer: tf.keras.optimizers.Optimizer = keras.optimizers.Adam(
                     learning_rate=.0001),
                 pre_trained: bool = True,
                 freeze_batch_norm: bool = False,
                 freeze_layers: bool = False,
                 base_model_type: ImageClassModels = ImageClassModels.MOBILENET_V2,
                 dense_layer_neurons: int = 1024,
                 dropout_rate: float = .5,
                 l1: float = 1e-4,
                 l2: float = 1e-4):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.optimizer = optimizer
        self.pre_trained = pre_trained
        self.freeze_layers = freeze_layers
        self.freeze_batch_norm = freeze_batch_norm
        self.dense_layer_neurons = dense_layer_neurons
        self.dropout_rate = dropout_rate
        self.l1 = l1
        self.l2 = l2
        self.set_base_model(base_model_type)

    def set_base_model(self, base_model_type: ImageClassModels):
        self.base_model_type = base_model_type
        self.base_model = self.base_model_type.value.model_func(
            weights='imagenet' if self.pre_trained else None,
            input_shape=self.input_shape,
            include_top=False
        )

    def create_model(self):
        if self.freeze_layers:
            self.base_model.trainable = False
        if self.freeze_batch_norm:
            for layer in self.base_model.layers:
                if isinstance(layer, keras.layers.BatchNormalization):
                    layer.trainable = False
        i = tf.keras.layers.Input([self.input_shape[0], self.input_shape[1], self.input_shape[2]], dtype=tf.float32)
        x = tf.cast(i, tf.float32)
        x = self.base_model_type.value.model_preprocessor(x)
        x = self.base_model(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(self.dense_layer_neurons, activation='relu',
                               kernel_regularizer=keras.regularizers.L1L2(l1=self.l1, l2=self.l2))(x)
        x = keras.layers.Dropout(self.dropout_rate)(x)
        output = keras.layers.Dense(self.n_classes, activation='softmax')(x)
        self.model = keras.Model(inputs=i, outputs=output)
        self.model.compile(
            optimizer=self.optimizer,
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy', 'categorical_crossentropy']
        )
        self.model.summary()
        return self.model

    def get_fine_tuning(self):
        print("self.model is found")
        self.base_model.trainable = True
        self.model.compile(
            optimizer=self.optimizer,
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy', 'categorical_crossentropy']
        )
        self.model.summary()
        return self.model

    def get_name(self):
        return f"{'pt-' if self.pre_trained else ''}{'fl-' if self.freeze_layers else ''}{'fbn-' if self.freeze_batch_norm else ''}" \
               f"{self.base_model_type.value.name}-d{self.dense_layer_neurons}-do{self.dropout_rate}" \
               f"{'-l1' + np.format_float_scientific(self.l1) if self.l1 > 0 else ''}{'-l2' + np.format_float_scientific(self.l2) if self.l2 > 0 else ''}" \
               f"-{random.randint(1111, 9999)}"
