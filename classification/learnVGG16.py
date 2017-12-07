from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, ZeroPadding2D
from keras.layers import Flatten, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import os
from tkinter import messagebox

batch_size = 25
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
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    'data/validation',
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical'
)

callbacks = list()
opt = Adam()
sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
fpath = 'models/weights.{epoch:02d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5'
cpcb = ModelCheckpoint(filepath=fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks.append(cpcb)

print('Make Model')
model = Sequential()

model.add(ZeroPadding2D((1, 1), input_shape=(128, 128, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print('Start Learn')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=90,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=90,
    callbacks=[cpcb]
)

total = 0
ok_total = 0
charName = ''
for dir in os.listdir('data/test'):
    dir1 = 'data/test/' + dir
    label = 0

    if dir == 'gavriel':
        label = 0
        charName = 'gavriel'

    elif dir == 'raphiel':
        label = 1
        charName = 'raphiel'

    elif dir == 'satanichia':
        label = 2
        charName = 'satanichia'

    elif dir == 'vigne':
        label = 3
        charName = 'vigne'

    for file in os.listdir(dir1):
        filepath = dir1 + '/' + file
        image = np.array(Image.open(filepath).resize((160, 160))).astype('float32')
        print(filepath)
        result = model.predict_classes(np.array([image / 255.]))
        print('label', label, 'result:', result[0])

        total += 1

        if label == result[0]:
            ok_total += 1
        else:
            im = Image.open(filepath)
            im.show()
            messagebox.showinfo('結果', 'この画像は' + charName + 'です')
            im.close()

print('Answer : ', ok_total / total * 100, '%')
