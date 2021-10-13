from PyQt5.QtCore import QThread, pyqtSignal
from pathlib import Path
from detector import App
import cv2

class CvtProcess(QThread):
    signal = pyqtSignal(str)
    def __init__(self, vpath, img_save_dir, lb_save_dir, filename, total):
        super().__init__()
        self.vpath = vpath
        self.img_save_dir = img_save_dir
        self.lb_save_dir = lb_save_dir
        self.filename = filename
        self.total = total
        self.detector = App()

    def __del__(self):
        self.wait()

    def run(self):
        self.cap = cv2.VideoCapture(self.vpath)
        cnt = 0
        while 1:
            _, frame = self.cap.read()
            if not _: break
            cnt += 1
            # if cnt % 250: continue
            if cnt % 25: continue
            self.signal.emit(f'processing: {cnt}/{self.total}')
            imgpath = f'{self.img_save_dir}/{self.filename}_{cnt:05}.jpg'
            lbpath = f'{self.lb_save_dir}/{self.filename}_{cnt:05}.txt'
            pred = self.detector.detect_img(frame, lbpath)
            cv2.imwrite(imgpath, frame)
        self.signal.emit(f'Done. Images saved at {self.img_save_dir}')

class AutoProcess(QThread):
    signal = pyqtSignal(str)
    def __init__(self, lb_save_dir, img_list):
        super().__init__()
        self.lb_save_dir = lb_save_dir
        self.img_list = img_list
        self.total = len(img_list)
        self.detector = App()

    def __del__(self):
        self.wait()

    def run(self):
        for i, img_path in enumerate(self.img_list):
            img = cv2.imread(img_path.as_posix())
            self.signal.emit(f'processing: {i+1}/{self.total}')
            lbpath = self.lb_save_dir / img_path.name.replace('.jpg', '.txt')
            pred = self.detector.detect_img(img, lbpath.as_posix())
        self.signal.emit(f'Done. Labels saved at {self.lb_save_dir}')

class Tool_Opt():
    def __init__(self):
        self.IMGSIZE = (int(1920 *0.7), int(1080 *0.7))
        self.IMG_DIR = './images'
        self.LABEL_DIR = './labels'
        self.VIDEO_DIR = './videos'
        for dirName in [self.IMG_DIR, self.LABEL_DIR, self.VIDEO_DIR]:
            Path(dirName).mkdir(exist_ok=True)
        self.IMG_LIST = []
        self.CUR_INDEX = 0
        self.CN = 4
        self.l_pic_bgc = (200, 200, 200)
        self.l_console_bgc = (180, 180, 180)