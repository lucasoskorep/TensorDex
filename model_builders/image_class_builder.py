from enum import Enum
from typing import Tuple

import tensorflow as tf
from tensorflow import keras

from .modelwrapper import ModelWrapper


class ImageClassModels(Enum):
    INCEPTION_V3 = ModelWrapper(
        keras.applications.InceptionV3,
        keras.applications.inception_v3.preprocess_input
    )
    XCEPTION = ModelWrapper(
        keras.applications.xception.Xception,
        keras.applications.inception_v3.preprocess_input
    )
    MOBILENET_V2 = ModelWrapper(
        keras.applications.mobilenet_v2.MobileNetV2,
        keras.applications.mobilenet_v2.preprocess_input
    )


class ImageClassModelBuilder(object):

    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 n_classes: int,
                 optimizer: tf.keras.optimizers.Optimizer = keras.optimizers.Adam(
                     learning_rate=.0001),
                 pre_trained: bool = True,
                 fine_tune: int = 0,
                 base_model: ImageClassModels = ImageClassModels.MOBILENET_V2):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.optimizer = optimizer
        self.pre_trained = pre_trained
        self.fine_tune = fine_tune
        self.base_model = base_model

    def set_base_model(self, base_model: ImageClassModels):
        self.base_model = base_model

    def create_model(self):

        base_model = self.base_model.value.model_func(
            weights='imagenet' if self.pre_trained else None,
            include_top=False
        )
        if self.pre_trained:
            if self.fine_tune > 0:
                for layer in base_model.layers[:-self.fine_tune]:
                    layer.trainable = False
            else:
                for layer in base_model.layers:
                    layer.trainable = False

        i = tf.keras.layers.Input([self.input_shape[0], self.input_shape[1], self.input_shape[2]], dtype=tf.float32)
        x = tf.cast(i, tf.float32)
        x = self.base_model.value.model_preprocessor(x)
        x = base_model(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-5))(x)
        x = keras.layers.Dropout(0.25)(x)
        output = keras.layers.Dense(self.n_classes, activation='softmax')(x)

        model = keras.Model(inputs=i, outputs=output)
        model.compile(optimizer=self.optimizer,
                      loss=keras.losses.CategoricalCrossentropy(),
                      metrics=[
                          'accuracy',
                          # 'mse'
                      ])
        model.summary()
        return model
