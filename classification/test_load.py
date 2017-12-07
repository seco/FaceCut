from keras.models import load_model
import numpy as np
from PIL import Image
import os
import tkinter as tk

root = tk.Tk()
root.withdraw()

model = load_model('models/weights.07-0.03-0.99-0.00-1.00.hdf5')

total = 0
ok_total = 0
name = ''
resDict = {0: 'gabriel', 1: 'raphiel', 2: 'satanichia', 3: 'vigne'}
for dir in os.listdir('data/test'):
    dir1 = 'data/test/' + dir
    label = 0

    if dir == 'gabriel':
        label = 0
        name = 'gabriel'
    elif dir == 'raphiel':
        label = 1
        name = 'raphiel'
    elif dir == 'satanichia':
        label = 2
        name = 'satanichia'
    elif dir == 'vigne':
        label = 3
        name = 'vigne'

    for file in os.listdir(dir1):
        filepath = dir1 + '/' + file
        image = np.array(Image.open(filepath).resize((160, 160))).astype('float32')
        result = model.predict_classes(np.array([image / 255.]))

        total += 1

        if label == result[0]:
            ok_total += 1
            print('正解')
            img = Image.open(filepath)
            img.show()
        else:
            print(filepath)
            img = Image.open(filepath)
            img.show()
            print('不正解')
            print('label:', label, 'result:', result[0])

print('Answer : ', ok_total / total * 100, '%')
