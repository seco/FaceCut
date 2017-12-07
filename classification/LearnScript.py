from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D
from keras.layers import Flatten, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import numpy as np
from PIL import Image
import os

image_list = list()
label_list = list()

for dir in os.listdir('data/validation'):
    dir1 = 'data/validation/' + dir
    label = 0

    if dir == 'gavriel':
        label = 0
    elif dir == 'raphiel':
        label = 1
    elif dir == 'satanichia':
        label = 2
    elif dir == 'vigne':
        label = 3

    for file in os.listdir(dir1):
        label_list.append(label)
        filepath = dir1 + '/' + file
        image = np.array(Image.open(filepath).resize((160, 160))).astype('float32')
        print(filepath)
        print(image.shape)
        image_list.append(image / 255)

image_list = np.array(image_list)
Y = to_categorical(label_list)
opt = Adam()
tbcb = TensorBoard(log_dir='logs', histogram_freq=1, write_graph=True, write_images=True)

print('Make Model')
model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', input_shape=(160, 160, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(4, kernel_initializer='uniform'))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print('Start Learn')
model.fit(image_list, Y, epochs=1000, batch_size=25, validation_split=0.1, callbacks=[tbcb])
model.save('models')

total = 0
ok_total = 0
for dir in os.listdir('data/test'):
    dir1 = 'data/test/' + dir
    label = 0

    if dir == 'gavriel':
        label = 0
    elif dir == 'raphiel':
        label = 1
    elif dir == 'satanichia':
        label = 2
    elif dir == 'vigne':
        label = 3

    for file in os.listdir(dir1):
        filepath = dir1 + '/' + file
        image = np.array(Image.open(filepath).resize((160, 160))).astype('float32')
        print(filepath)
        result = model.predict_classes(np.array([image / 255.]))
        print('label', label, 'result:', result[0])

        total += 1

        if label == result[0]:
            ok_total += 1

print('Answer : ', ok_total / total * 100, '%')


