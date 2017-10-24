# import modules
import cv2
import os
import re
import sys
import tkinter as tk
from tkinter import filedialog, messagebox


# This class can cut the anime face
# You must put the file 'lbpcascade_animeface.xml' in the current folder
class Cutter:

    # initialize the variables
    def __init__(self):
        self.cascade_path = 'lbpcascade_animeface.xml'
        self.video_paths = list()
        self.save_path = ''
        self.IsChoice = None
        self.frame = None
        self.cap = None
        self.num = 0

    # select the Video folder
    def select_make(self):
        self.IsChoice = True
        root = tk.Tk()
        root.withdraw()
        dir_path = filedialog.askdirectory(initialdir='C:/', title='動画フォルダを選択してください')
        paths = list()
        try:
            for i, filename in enumerate(os.listdir(dir_path)):
                fn = str(dir_path + '/' + filename)
                self.video_paths.append(fn)
                path, ext = os.path.splitext(filename)
                if not ext == '.mp4' or not ext == '.avi':
                    continue
                paths = [(re.search("[0-9]+", x).group(), x) for x in self.video_paths]
            paths.sort(key=lambda x: int(x[0]))
            for i, p in enumerate(paths):
                self.video_paths[i] = p[1]
            del root
        except:
            choice = messagebox.askquestion('Error : Can\'t find file', '動画ファイルが見つかりませんでした\n'
                                                                        '終了しますか？')
            if choice == 'yes':
                sys.exit()
        if len(self.video_paths) == 0:
            self.IsChoice = False

    # select the Save folder
    def select_save(self):
        root = tk.Tk()
        root.withdraw()
        self.save_path = filedialog.askdirectory(initialdir='C:/', title='保存フォルダを選択してください')
        if self.save_path == '':
            choice = messagebox.askquestion('Error', '保存先が選択されていません\n'
                                                     '終了しますか？')
            if choice == 'yes':
                sys.exit()
        self.IsChoice = True

    # script of cut face
    def cut(self, filename):

        # change the path to use easily
        def change(p):
            result = ''
            for w in p:
                if w == '\\':
                    result += '/'
                else:
                    result += w
            return result

        # cut and return face
        def face_cut():
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            cascade = cv2.CascadeClassifier(self.cascade_path)
            facerect = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))

            print(facerect)
            return facerect

        self.cap = cv2.VideoCapture(filename)
        self.num += 1
        frame_num = 0
        save_num = 1
        other_num = 0

        # if the directory you select has not faces and other directory, make those directory
        output_dir = change(os.path.abspath(self.save_path))
        if not os.path.exists(output_dir + '/faces'):
            os.mkdir(output_dir + '/faces')
        output_dir += '/faces'
        output_dir = change(output_dir)
        other_dir = change(os.path.abspath(self.save_path))
        if not os.path.exists(other_dir + '/other'):
            os.mkdir(other_dir + '/other')
        other_dir += '/other'

        # cut and save img loop
        while self.cap.isOpened():

            frame_num += 1
            ret, self.frame = self.cap.read()
            if not ret:
                break
            if frame_num % 50 == 0:
                face = face_cut()

                # if img has not anime face, save the other directory
                if len(face) == 0:
                    other_path = os.path.join(other_dir, 'd_{0}_{1}.jpg'.format(self.num, other_num))
                    other_num += 1
                    try:
                        cv2.imwrite(other_path, self.frame)
                    except:
                        messagebox.showerror('Error', '保存に失敗しました')

                # save the img which has the anime face (Size : 28x28)
                for j, (x, y, w, h) in enumerate(face):
                    face_img = self.frame[y: y + h, x: x + w]
                    # face_img = cv2.resize(face_img, (28, 28))
                    output_path = str(output_dir) + '/' + '{0}_{1}_{2}.jpg'.format(self.num, save_num, j)
                    save_num += 1
                    try:
                        cv2.imwrite(output_path, face_img)
                        print(output_path)
                    except:
                        messagebox.showerror('Error', '保存に失敗しました')

        self.cap.release()

    # main routine
    def run(self):
        # select the directory which has video files
        while not self.IsChoice:
            self.select_make()
        self.IsChoice = False
        # select the save directory
        while not self.IsChoice:
            self.select_save()
        # cut the anime face
        for fn in self.video_paths:
            self.cut(fn)
        del self


# main function
if __name__ == '__main__':
    cutter = Cutter()
    cutter.run()
