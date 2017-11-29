from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers import Convolution2D, Flatten, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os


class Model:
    def __init__(self):
        self.model = Sequential()
        self.opt = Adam()
        self.Y = None
        self.image_list = list()
        self.label_list = list()

    def _add(self):
        self.model.add(Convolution2D(32, 3, 3, padding='same', input_shape=(3, 100, 100)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(32, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())

        self.model.add(Dense(200))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(3))
        self.model.add(Activation('softmax'))

    def _compile(self):
        self.model.compile(loss='categorical_crossentropy', optimizer=self.opt, metrics=['accuracy'])
        self.Y = to_categorical(self.label_list)

    def _fit(self):
        self.image_list, self.label_list = self.set_image()
        self._add()
        self._compile()
        self.model.fit(self.image_list, self.Y, nb_epoch=1000, batch_size=25, validation_split=0.1)

    @staticmethod
    def set_image():
        image_list = list()
        label_list = list()
        for dir in os.listdir('data/train'):
            dir1 = "data/train/" + dir
            label = 0

            if dir == 'spam':
                label = 0
            elif dir == 'ham':
                label = 1
            elif dir == 'egg':
                label = 2

            for file in os.listdir(dir1):
                label_list.append(label)
                filepath = dir1 + '/' + file
                image = np.array(Image.open(filepath).resize(100, 100))
                print(filepath)
                image_list.append(image / 255.)

        image_list = np.array(image_list)
        return image_list, label_list

    def _learn(self):
        total = 0
        ok_total = 0
        for dir in os.listdir('data/test'):
            dir1 = 'data/test/' + dir
            label = 0

            if dir == 'spam':
                label = 0
            elif dir == 'ham':
                label = 1
            elif dir == 'egg':
                label = 2

            for file in os.listdir(dir1):
                filepath = dir1 + '/' + file
                image = np.array(Image.open(filepath).resize(100, 100))
                print(filepath)
                result = self.model.predict_classes(np.array([image / 255.]))
                print('label:', label, 'result:', result[0])

                total += 1

                if label == result[0]:
                    ok_total += 1

        print('Answer : ', ok_total / total * 100, '%')

    def run(self):
        self._fit()
        self._learn()


if __name__ == '__main__':
    model = Model()
    model.run()


