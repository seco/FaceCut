from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers import Convolution2D, Flatten, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os


class Model:
    def __init__(self, image_list, label_list):
        self.model = Sequential()
        self.opt = Adam()
        self.image_list = image_list
        self.Y = to_categorical(label_list)

    def add(self):
        self.model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, 100, 100)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(32, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())

        self.model.add(Dense(200))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(3))
        self.model.add(Activation('softmax'))

    def compile(self):
        self.model.compile(loss='categorical_crossentropy', optimizer=self.opt, metrics=['accuracy'])

    def fit(self):
        self.add()
        self.compile()
        self.model.fit(self.image_list, self.Y, nb_epoch=1000, batch_size=25, validation_split=0.1)


