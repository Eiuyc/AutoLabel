from PyQt5.QtCore import QThread, pyqtSignal
from pathlib import Path
from detector import App
from cv2 import VideoCapture, imread, imwrite
from PyQt5.QtCore import Qt

class CvtProcess(QThread):
    signal = pyqtSignal(str)
    def __init__(self, vpath, img_save_dir, lb_save_dir, filename, total, mode):
        super().__init__()
        self.mode = mode
        self.vpath = vpath
        self.img_save_dir = img_save_dir
        self.lb_save_dir = lb_save_dir
        self.filename = filename
        self.total = total
        self.detector = App(self.mode)

    def __del__(self):
        self.wait()

    def run(self):
        self.cap = VideoCapture(self.vpath)
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
            imwrite(imgpath, frame)
        self.signal.emit(f'Done. Images saved at {self.img_save_dir}')

class AutoProcess(QThread):
    signal = pyqtSignal(str)
    def __init__(self, lb_save_dir, img_list, mode):
        super().__init__()
        self.mode = mode
        self.lb_save_dir = lb_save_dir
        self.img_list = img_list
        self.total = len(img_list)
        self.detector = None

    def __del__(self):
        self.wait()

    def run(self):
        if self.detector is None:
            self.detector = App(self.mode)
        for i, img_path in enumerate(self.img_list):
            img = imread(img_path.as_posix())
            self.signal.emit(f'processing: {i+1}/{self.total}')
            lbpath = self.lb_save_dir / img_path.name.replace('.jpg', '.txt')
            pred = self.detector.detect_img(img, lbpath.as_posix())
        self.signal.emit(f'Done. Labels saved at {self.lb_save_dir}')

class CutProcess(QThread):
    signal = pyqtSignal(str)
    def __init__(self, obj_save_dir, img_path, lbs):
        super().__init__()
        self.obj_save_dir = obj_save_dir
        self.img_path = img_path
        self.lbs = lbs
        self.total = len(lbs)

    def __del__(self):
        self.wait()

    def run(self):
        img = imread(self.img_path.as_posix())
        utils = Utils()
        lbs = list(map(utils.cxyxy2cxywh, self.lbs))
        lbs = list(map(lambda x: utils.cxywh2cxyxy(x, img.shape[:2][::-1]), lbs))
        cuts = {}
        for i, lb in enumerate(lbs):
            self.signal.emit(f'cutting: {i+1}/{self.total}')
            c, x1, y1, x2, y2 = map(int, lb)
            cut = img[y1:y2, x1:x2]
            try:
                cuts[c].append(cut)
            except:
                cuts[c] = [cut]
        i = 0
        for c in cuts.keys():
            c_dir = self.obj_save_dir / str(c)
            c_dir.mkdir(exist_ok=1)
            for cut in cuts[c]:
                i += 1
                self.signal.emit(f'cutting: {i}/{self.total}')
                obj_save_path = c_dir / f'{c}_{i}_{self.img_path.name}'
                imwrite(obj_save_path.as_posix(), cut)
            
        self.signal.emit(f'Done. Cuts saved at {self.obj_save_dir}')

class Tool_Opt():
    def __init__(self):
        self.IMGSIZE = (int(1920 *0.7), int(1080 *0.7))
        self.IMG_DIR = './images'
        self.LABEL_DIR = './labels'
        self.VIDEO_DIR = './videos'
        self.CUT_DIR = './cuts'
        for dirName in [self.IMG_DIR, self.LABEL_DIR, self.VIDEO_DIR, self.CUT_DIR]:
            Path(dirName).mkdir(exist_ok=True)
        self.IMG_LIST = []
        self.CUR_INDEX = 0
        self.CN = 7
        self.l_pic_bgc = (200, 200, 200)
        self.l_console_bgc = (180, 180, 180)

class Review_Opt():
    def __init__(self):
        self.IMGSIZE = (int(1920 *0.7), int(1080 *0.7))
        self.LB_DIR = None
        self.IMG_DIR = Path('./images')
        self.IMG_LIST = []
        self.CUR_INDEX = 0
        self.CN = 7
        self.NUM_keys = [Qt.Key_0, Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5, Qt.Key_6]
        self.l_pic_bgc = (200, 200, 200)
        self.l_console_bgc = (180, 180, 180)

class Utils():
    def __init__(self, IMGSIZE=(int(1920 *0.7), int(1080 *0.7))):
        self.size = IMGSIZE

    def cxyxy2cxywh(self, cxyxy, size=None):
        if not size: size = self.size
        c = cxyxy[0]
        w = cxyxy[3] - cxyxy[1]
        h = cxyxy[4] - cxyxy[2]
        x = cxyxy[1] + w/2
        y = cxyxy[2] + h/2
        cxywh = [c, x/size[0], y/size[1], w/size[0], h/size[1]]
        return cxywh

    def cxywh2cxyxy(self, cxywh, size=None):
        if not size: size = self.size
        c, x, y, w, h = cxywh
        x *= size[0]; w *= size[0]
        y *= size[1]; h *= size[1]
        x0 = x - w/2
        y0 = y - h/2
        x1 = x + w/2
        y1 = y + h/2
        cxyxy = [c, x0, y0, x1, y1]
        return cxyxy
