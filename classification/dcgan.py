from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D
from keras.layers import Flatten, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers import BatchNormalization
from keras.layers.convolutional import UpSampling2D
import numpy as np
import sys
import os
from PIL import Image


class dcgan:

    def __init__(self):
        self.discriminator = Sequential()
        self.generator = Sequential()
        self.image_list = None

    def make_discriminator(self):
        self.discriminator.add(Conv2D(64, (3, 3), padding='same', strides=(2, 2), input_shape=[32, 32, 3]))
        self.discriminator.add(LeakyReLU())
        self.discriminator.add(Conv2D(128, (3, 3), padding='same', strides=(2, 2)))
        self.discriminator.add(BatchNormalization())
        self.discriminator.add(LeakyReLU())
        self.discriminator.add(Conv2D(256, (3, 3), padding=(3, 3), strides=(2, 2)))
        self.discriminator.add(BatchNormalization())
        self.discriminator.add(LeakyReLU())
        self.discriminator.add(Flatten())
        self.discriminator.add(Dense(2048))
        self.discriminator.add(BatchNormalization())
        self.discriminator.add(LeakyReLU())
        self.discriminator.add(Dense(1, activation='sigmoid'))

    def make_generator(self):
        self.generator.add(Conv2D(64, (3, 3), padding='same', input_shape=(4, 4, 4)))
        self.generator.add(UpSampling2D())
        self.generator.add(Conv2D(128, (3, 3), padding='same'))
        self.generator.add(BatchNormalization())
        self.generator.add(ELU())
        self.generator.add(UpSampling2D())
        self.generator.add(Conv2D(128, (3, 3), padding='same'))
        self.generator.add(BatchNormalization())
        self.generator.add(ELU())
        self.generator.add(UpSampling2D())
        self.generator.add(Conv2D(1, (5, 5), padding='same', activation='tanh'))

    def compile(self):
        opt_dis = Adam(lr=1e-5, beta_1=0.1)
        self.discriminator.compile(optimizer=opt_dis,
                                   loss='binary_crossentropy',
                                   metrics=['accuracy'])
        self.discriminator.trainable = False
        dcgan = Sequential([self.generator, self.discriminator])
        opt_gen = Adam(lr=2e-4, beta_1=0.5)
        dcgan.compile(optimizer=opt_gen,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    def learn(self):
        def create_random_features(num):
            return np.random.uniform(low=-1, high=1,
                                     size=[num, 4, 4, 4])
        for epoch in range(1, sys.maxsize):

            print('epoch : ')

    def create_batch(self):
        dir = 'Manydata'

        for file in os.listdir(dir):
            filepath = dir + '/' + file
            image = np.array(Image.open(filepath).resize((160, 160))).astype('float32')
            print(filepath)
            print(image.shape)
            self.image_list.append(image / 255)
