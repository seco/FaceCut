from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers import BatchNormalization
from keras.layers.convolutional import UpSampling2D
import numpy as np
import sys
import os
from PIL import Image
import time
import math


class dcgan:

    def __init__(self):
        self.discriminator = Sequential()
        self.generator = Sequential()
        self.dcgan = None
        self.image_list = list()

    def make_discriminator(self):
        print('create discriminator')
        self.discriminator.add(Conv2D(64, (3, 3), padding='same', strides=(2, 2), input_shape=[160, 160, 3]))
        self.discriminator.add(LeakyReLU())
        self.discriminator.add(Conv2D(128, (3, 3), padding='same', strides=(2, 2)))
        self.discriminator.add(BatchNormalization())
        self.discriminator.add(LeakyReLU())
        self.discriminator.add(Conv2D(256, (3, 3), padding='same', strides=(2, 2)))
        self.discriminator.add(BatchNormalization())
        self.discriminator.add(LeakyReLU())
        self.discriminator.add(Flatten())
        self.discriminator.add(Dense(2048))
        self.discriminator.add(BatchNormalization())
        self.discriminator.add(LeakyReLU())
        self.discriminator.add(Dense(1, activation='sigmoid'))

    def make_generator(self):
        print('create generator')
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

    def createImageList(self):
        print('create ImageList')
        dir = 'ManyData'
        for file in os.listdir(dir):
            filepath = dir + '/' + file
            image = np.array(Image.open(filepath).resize((160, 160))).astype('float32')
            self.image_list.append(image / 255)

    @staticmethod
    def set_trainable(model, trainable):
        print('set trainable')
        model.trainable = trainable
        for layer in model.layers:
            layer.trainable = trainable

    def compile(self):
        print('compile the model')
        self.make_generator()
        self.make_discriminator()
        opt_dis = Adam(lr=1e-5, beta_1=0.1)
        self.discriminator.compile(optimizer=opt_dis,
                                   loss='binary_crossentropy',
                                   metrics=['accuracy'])
        self.discriminator.trainable = False
        self.set_trainable(model=self.discriminator, trainable=False)
        self.dcgan = Sequential([self.generator, self.discriminator])
        opt_gen = Adam(lr=2e-4, beta_1=0.5)
        self.dcgan.compile(optimizer=opt_gen,
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def combine(self, generated):
        print('generated image')
        num = generated.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
        shape = generated.shape[1:3]
        image = np.zeros((height*shape[0], width*shape[1]), dtype=generated.dtype)
        for index, img in enumerate(generated):
            i = int(index / width)
            j = index % index
            image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img[:, :, 0]
            return image

    def learn(self):

        def create_random_features(num):
            return np.random.uniform(low=-1, high=1,
                                     size=[num, 4, 4, 4])

        batch_size = 200
        wait = 0
        test_num = 1000
        rnd_test = create_random_features(test_num)
        faked_curve = np.zeros([0, 2])
        met_curve = np.zeros([0, 4])
        start = time.time()
        loss_g = acc_g = loss_d = acc_d = None
        self.compile()
        for epoch in range(1, sys.maxsize):
            print('epoch : {0}'.format(epoch))

            np.random.shuffle(self.image_list)
            rnd = create_random_features(len(self.image_list))
            for i in range(int(math.ceil(len(self.image_list) / batch_size))):
                print('batch : ', i, end='\r')
                X_batch = self.image_list[i*batch_size:(i+1)*batch_size]
                rnd_batch = rnd[i*batch_size:(i+1)*batch_size]

                loss_g, acc_g = self.dcgan.train_on_batch(rnd_batch, [0]*len(rnd_batch))
                generated = self.generator.predict(rnd_batch)
                if epoch % 20 == 0:
                    image = self.combine(generated)
                    image = image*127.5 + 127.5
                    dir1 = 'generated/' + str(epoch) + '_' + str(i) + '.png'
                    Image.fromarray(image.astype(np.uint8)).save(dir1)
                X = np.append(X_batch, generated, axis=0)
                y = [0] * len(X_batch) + [1] * len(generated)
                loss_d, acc_d = self.discriminator.train_on_batch(X, y)

                met_curve = np.append(met_curve, [[loss_d, acc_d, loss_g, acc_g]], axis=0)
            val_loss, faked = self.dcgan.evaluate(rnd_test, [0] * test_num)
            print('epoch end:')
            print('d: loss: {0:.3e} acc: {1:.3f}'.format(loss_d, acc_d))
            print('g: loss: {0:.3e} acc: {1:.3f}'.format(loss_g, acc_g))
            print('faked: {0}'.format(faked))

            faked_curve = np.append(faked_curve, [[val_loss, faked]], axis=0)
            np.save('faked_curve', faked_curve)
            np.save('met_curve', met_curve)

            if epoch % 10 == 0:
                model = Sequential([self.generator, self.discriminator])
                model.save('models/{0}.hdf5'.format(epoch))
                print('save : {0}.hdf5'.format(epoch))
            print('')

            if faked == 0 or faked == 1:
                wait += 1
                if wait > 50:
                    print('wait reach 50')
                    print('elapsed time : {0}sec'.format(time.time()-start))
                    exit(0)
                else:
                    wait = 0

    def run(self):
        self.createImageList()
        self.compile()
        self.learn()


if __name__ == '__main__':
    gan = dcgan()
    gan.run()
