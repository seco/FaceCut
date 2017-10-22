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

    # initialize
    def __init__(self):
        self.cascade_path = 'lbpcascade_animeface.xml'
        self.video_paths = list()
        self.save_path = ''
        self.IsChoice = None
        self.frame = None
        self.cap = None

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

    # change the path to use easily
    @staticmethod
    def change(p):
        path = ''
        for w in p:
            if w == '\\':
                path += '/'
            else:
                path += w
        return path

    # cut and return face
    def face_cut(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        cascade = cv2.CascadeClassifier(self.cascade_path)
        facerect = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))

        print(facerect)
        return facerect

    # script of cut face
    def cut(self, filename):
        self.cap = cv2.VideoCapture(filename)
        frame_num = 0
        save_num = 1
        path, ext = os.path.splitext(os.path.basename(filename))
        other_num = 0

        output_dir = self.change(os.path.abspath(self.save_path))
        if not os.path.exists(output_dir + '/faces'):
            os.mkdir(output_dir + '/faces')
            output_dir += '/faces'
        output_dir = self.change(output_dir)
        other_dir = self.change(os.path.abspath(self.save_path))
        if not os.path.exists(other_dir + '/other'):
            os.mkdir(other_dir + '/other')
            other_dir += '/other'

        while self.cap.isOpened():

            frame_num += 1
            ret, self.frame = self.cap.read()
            if frame_num % 50 == 0:
                face = self.face_cut()
                if len(face) == 0:
                    other_path = os.path.join(other_dir, 'd_{0}_{1}.jpg'.format(path, other_num))
                    other_num += 1
                    try:
                        cv2.imwrite(other_path, self.frame)
                    except:
                        messagebox.showerror('Error', '保存に失敗しました')

                for j, (x, y, w, h) in enumerate(face):
                    face_img = self.frame[y: y + h, x: x + w]
                    face_img = cv2.resize(face_img, (50, 50))
                    output_path = str(output_dir) + '/' + '{0}_{1}_{2}.jpg'.format(path, save_num, j)
                    save_num += 1
                    try:
                        cv2.imwrite(output_path, face_img)
                        print(output_path)
                    except:
                        messagebox.showerror('Error', '保存に失敗しました')

        self.cap.release()

    # main routine
    def run(self):
        while True:
            self.select_make()
            if self.IsChoice:
                break
        while True:
            self.select_save()
            if self.IsChoice:
                break
        for fn in self.video_paths:
            self.cut(fn)
        del self


if __name__ == '__main__':
    cutter = Cutter()
    cutter.run()
