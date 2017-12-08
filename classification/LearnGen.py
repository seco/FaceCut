from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D
from keras.layers import Flatten, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import os

batch_size = 32
epochs = 50

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(160, 160),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    'data/validation',
    target_size=(160, 160),
    batch_size=batch_size,
    class_mode='categorical'
)

callbacks = list()
opt = Adam()
fpath = 'models/weights.{epoch:02d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5'
tbcb = TensorBoard(log_dir='logs', histogram_freq=1)
cpcb = ModelCheckpoint(filepath=fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
#callbacks.append(tbcb)
callbacks.append(cpcb)

print('Make Model')
model = Sequential()

model.add(Conv2D(96, (3, 3), padding='same', input_shape=(160, 160, 3)))
model.add(Activation('relu'))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(11, kernel_initializer='uniform'))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print('Start Learn')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=90,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=90,
    callbacks=callbacks
)

total = 0
ok_total = 0
charName = ''
for dir in os.listdir('data/test'):
    dir1 = 'data/test/' + dir
    label = 0

    if dir == 'dagashiya':
        label = 0
    elif dir == 'gabriel':
        label = 1
    elif dir == 'hikanee':
        label = 2
    elif dir == 'hotaru':
        label = 3
    elif dir == 'komari':
        label = 4
    elif dir == 'konomi':
        label = 5
    elif dir == 'nattsun':
        label = 6
    elif dir == 'raphiel':
        label = 7
    elif dir == 'renge':
        label = 8
    elif dir == 'satanichia':
        label = 9
    elif dir == 'vigne':
        label = 10

    for file in os.listdir(dir1):
        filepath = dir1 + '/' + file
        image = np.array(Image.open(filepath).resize((160, 160))).astype('float32')
        print(filepath)
        result = model.predict_classes(np.array([image / 255.]))
        print('label', label, 'result:', result[0])

        total += 1

        if label == result[0]:
            ok_total += 1
            print('正解')
        else:
            print(filepath)
            print('不正解')

print('Answer : ', ok_total / total * 100, '%')