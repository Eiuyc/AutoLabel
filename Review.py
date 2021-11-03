import os, sys, cv2, base64, argparse
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QComboBox
from PyQt5.QtWidgets import QFileDialog, QLabel, QTextBrowser, QRadioButton
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QIcon
from PyQt5.QtGui import QTextCursor, QPixmap
from PyQt5.QtCore import QRect, Qt

from Modules import Review_Opt, Utils

class MyLabel(QLabel):
    def __init__(self, p, window):
        super().__init__(p)
        self.labels = []
        self.press = False
        self.move = False
        self.x0 = 0
        self.y0 = 0
        self.x1 = 0
        self.y1 = 0
        self.window = window
        # self.colors = [Qt.red, Qt.green, Qt.blue, Qt.yellow]
        self.colors = [
            QColor(255, 0, 0),
            QColor(0, 255, 0),
            QColor(0, 0, 255),
            QColor(255, 255, 0),
            QColor(255, 0, 255),
            QColor(0, 255, 255),
            QColor(255, 255, 255),
        ]
        # self.boarder = 0.6
        self.boarder = 1
        self.linestyle = Qt.SolidLine

    def mousePressEvent(self, evt):
        self.press = True
        if self.window.btn_mode_ist.isChecked():
            self.x0 = evt.x()
            self.y0 = evt.y()
            self.x1 = evt.x()
            self.y1 = evt.y()
        elif self.window.btn_mode_del.isChecked():
            self.window.delete_boxes(evt.x(), evt.y())
        elif self.window.btn_mode_chg.isChecked():
            self.window.change_boxes(evt.x(), evt.y())
        self.update()

    def mouseReleaseEvent(self,evt):
        self.press = False
        if self.window.btn_mode_ist.isChecked() and self.move:
            self.move = False
            rect = [
                min(self.x0, self.x1),
                min(self.y0, self.y1),
                max(self.x0, self.x1),
                max(self.y0, self.y1)
            ]
            cxyxy = [int(self.window.edit_cls.currentText()),] + rect
            self.labels.append(cxyxy)
            self.window.log(('Append [%d' + ' %d' * 4) % tuple(cxyxy) + '].')
            self.window.log_length()

    def mouseMoveEvent(self, evt):
        if self.press:
            self.move = True
            if self.window.btn_mode_ist.isChecked():
                self.x1 = evt.x()
                self.y1 = evt.y()
            self.update()
    
    def get_rect_color(self, c):
        return self.colors[int(c)]

    def paintEvent(self, evt):
        super().paintEvent(evt)
        if not self.window.view_state: return
        painter = QPainter(self)
        painter.setFont(QFont('Decorative', 10))
        for c, x0, y0, x1, y1 in self.labels:
            rect = QRect(int(x0), int(y0), int(x1 - x0), int(y1 - y0))
            painter.setPen(QPen(self.get_rect_color(c), self.boarder, self.linestyle))
            painter.drawRect(rect)
            painter.drawText(rect, Qt.AlignCenter, str(c))
        current_c = self.window.edit_cls.currentText()
        painter.setPen(QPen(self.get_rect_color(current_c), self.boarder, self.linestyle))
        if self.press and self.window.btn_mode_ist.isChecked():
            rect = QRect(self.x0, self.y0, self.x1 - self.x0, self.y1 - self.y0)
            painter.drawRect(rect)
            painter.drawText(rect, Qt.AlignCenter, current_c)

class Eiuyc_Label_Tool(QWidget):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.opt = Review_Opt()
        self.utils = Utils(self.opt.IMGSIZE)
        self.view_state = True
        self.l_pic = MyLabel(self, self)
        self.l_pic.setGeometry(0, 0, *self.opt.IMGSIZE)
        self.l_pic.setStyleSheet(f'background-color: rgb{self.opt.l_pic_bgc}')

        self.l_console = QTextBrowser(self)
        self.l_console.setGeometry(self.opt.IMGSIZE[0], 0, 500, self.opt.IMGSIZE[1] + 100)
        self.l_console.setText('Label tool for yolov5.\n\n=======')
        self.l_console.setStyleSheet(f'background-color: rgb{self.opt.l_console_bgc}')
        self.l_console.setAlignment(Qt.AlignLeft|Qt.AlignTop)

        btn_load = QPushButton(self)
        btn_load.setText('load')
        btn_load.move(30,self.opt.IMGSIZE[1]+20)
        btn_load.clicked.connect(self.func_load)
        btn_load.setToolTip('Load a directory which contains images')

        btn_prev = QPushButton(self)
        btn_prev.setText('<---')
        btn_prev.move(30+200,self.opt.IMGSIZE[1]+20)
        btn_prev.clicked.connect(self.func_prev)
        btn_prev.setToolTip('Show previous image')

        btn_next = QPushButton(self)
        btn_next.setText('--->')
        btn_next.move(30+200*2,self.opt.IMGSIZE[1]+20)
        btn_next.clicked.connect(self.func_next)
        btn_next.setToolTip('Show next image')

        btn_remv = QPushButton(self)
        btn_remv.setText('remv')
        btn_remv.move(30+200*3,self.opt.IMGSIZE[1]+20)
        btn_remv.clicked.connect(self.func_remv)
        btn_remv.setToolTip('Remove the last box of the box list')

        btn_save = QPushButton(self)
        btn_save.setText('save')
        btn_save.move(30+200*4,self.opt.IMGSIZE[1]+20)
        btn_save.clicked.connect(self.func_save)
        btn_save.setToolTip('Save the label in a txt file')

        l_cls = QLabel(self)
        l_cls.setText('class id:')
        l_cls.move(-50+200*4,self.opt.IMGSIZE[1]+62)

        self.edit_cls = QComboBox(self)
        self.edit_cls.addItems([str(i) for i in range(self.opt.CN)])
        self.edit_cls.setGeometry(31+200*4,self.opt.IMGSIZE[1]+60, 91,20)
        
        l_mode = QLabel(self)
        l_mode.setText('mode:')
        l_mode.move(30+200*5,self.opt.IMGSIZE[1]+25)

        self.btn_mode_ist = QRadioButton('insert', self)
        self.btn_mode_ist.setChecked(True)
        self.btn_mode_ist.move(30+200*5+50,self.opt.IMGSIZE[1]+25)
        self.btn_mode_ist.setToolTip('Click and drag mouse to draw boxes')

        self.btn_mode_del = QRadioButton('delete', self)
        self.btn_mode_del.move(30+200*5+50,self.opt.IMGSIZE[1]+50)
        self.btn_mode_del.setToolTip('Click mouse to delete boxes')
        
        self.btn_mode_chg = QRadioButton('change', self)
        self.btn_mode_chg.move(30+200*5+50,self.opt.IMGSIZE[1]+75)
        self.btn_mode_chg.setToolTip('Click mouse to change label with selected class id')
        
        pm = QPixmap()
        pm.loadFromData(base64.b64decode('AAABAAEAICAAAAEAIACoEAAAFgAAACgAAAAgAAAAQAAAAAEAIAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAbHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xkbG/8YGhr/GRsb/xocHP8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/0lKSv+QkZH/uLm5/8PExP+5urr/jo6O/0ZHR/8bHBz/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/ycoKP+xsbH//Pz8///////09PT/vr6+/6Wlpf+ys7P/4uPj/7a2tv8zNTX/Gxwc/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8aHBz/v8DA///////+/v7/ubm5/ywuLv8ZGxv/Gx0d/xocHP8aGxv/VVZW/5WWlv8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/1NTU//+/v7//////9jY2P8eICD/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8aHBz/GRsb/xocHP8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/e3x8////////////hYWF/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf9vcHD//v7+//////90dHT/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/ykqKv/p6en//////62trf8ZGxv/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Ghwc/0JDQ//Pz8//+/v7/3N0dP8dHx//GBoa/xcZGf8aHBz/Ghwc/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xweHv9QUFD/np6e/9zc3P+8vb3/vr+//9rb2/9VVlb/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/MjQ0/7Ozs//w8PD/h4eH/0xMTP9ERUX/VVVV/zs9Pf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/yAiIv/W1tb//////4CBgf8aHBz/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/UFFR//7+/v/+/v7/P0BA/xsdHf8bHR3/Gx0d/xsdHf8cHR3/HiAg/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf9KS0v//v7+//7+/v9LS0v/Gx0d/xsdHf8bHR3/Gx0d/0RFRf+bm5v/Ghsb/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xweHv/Hx8f//////7Ozs/8dHh7/Gx0d/xsdHf8ZGxv/np6e//n5+f88Pj7/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/ykqKv+goKD/6+vr/8zMzP+BgoL/f39//7+/v//p6en/nJyc/ygqKv8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xkbG/8jJCT/Pj8//01PT/9PUFD/P0BA/yIkJP8ZGxv/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/Gx0d/xsdHf8bHR3/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA='))
        icon = QIcon()
        icon.addPixmap(pm, QIcon.Normal, QIcon.Off)
        self.setWindowIcon(icon)
        self.setGeometry(100, 100, self.opt.IMGSIZE[0]+500, self.opt.IMGSIZE[1]+100)
        self.setWindowTitle('Label tool')
        self.grabKeyboard()  # declare manually in multiple widgets case to listening non-functional keys
        self.show()

    def log(self, msg):
        self.l_console.append('>>>' + msg + '\n')

    def log_length(self):
        self.log(f'Current box amount is {len(self.l_pic.labels)}.')

    def log_process(self, msg):
        self.l_console.moveCursor(QTextCursor.End, QTextCursor.MoveAnchor)
        self.l_console.moveCursor(QTextCursor.StartOfLine, QTextCursor.MoveAnchor)
        lastline = self.l_console.textCursor()
        lastline.select(QTextCursor.LineUnderCursor)
        lastline.removeSelectedText()
        lastline.deletePreviousChar()
        self.l_console.append(msg)

    def func_load(self):
        self.opt.LB_DIR = Path(QFileDialog.getExistingDirectory(self, 'choose a directory', ''))
        files = []
        if self.opt.LB_DIR.exists():
            files = list(self.opt.LB_DIR.glob('*.txt'))

        self.opt.LB_LIST = files
        if not len(self.opt.LB_LIST):
            self.log('No availabel images.')
            return
        if len(self.opt.LB_LIST) < 5:
            msg = ', '.join(self.opt.LB_LIST)
        else:
            msg = f'{self.opt.LB_LIST[0]}, {self.opt.LB_LIST[1]}, ..., {self.opt.LB_LIST[-1]}'
        self.log(f'found {len(self.opt.LB_LIST)} labels:')
        self.log('[' + msg + ']')
        self.opt.IMG_DIR = self.opt.LB_DIR.resolve().parents[1] / 'images'
        self.opt.CUR_INDEX = 0
        self.show_img()

    def func_prev(self):
        self.opt.CUR_INDEX -= 1
        if not self.check_index(): return
        self.show_img()

    def func_next(self):
        self.opt.CUR_INDEX += 1
        if not self.check_index(): return
        self.show_img()

    def show_img(self):
        image_path = self.get_image_path().as_posix()
        self.log('Current image path is:\n'+ image_path)
        img = QPixmap(image_path).scaled(*self.opt.IMGSIZE)
        self.l_pic.setPixmap(img)
        self.initial_label()

    def func_save(self):
        if not self.check_index():
            self.log('Save failed.')
            return
        label_path = self.get_label_path()
        image_path = self.get_image_path()
        with open(label_path, 'w') as f:
            f.writelines(map(lambda x: ('%d'+' %f'*4) % tuple(self.utils.cxyxy2cxywh(x)) +'\n', self.l_pic.labels))
        self.log(f'Label saved to {label_path}.')

    def func_remv(self):
        if self.l_pic.labels:
            self.l_pic.labels.remove(self.l_pic.labels[-1])
            self.log('Remove the last box.')
            self.log_length()
            self.l_pic.update()
            return
        self.log('Nothing to do.')

    def check_index(self):
        if not self.opt.LB_LIST:
            self.log('No availabel image.')
            self.opt.CUR_INDEX = 0
            return False
        if self.opt.CUR_INDEX < 0:
            self.opt.CUR_INDEX = 0
            self.log('This is the first one.')
            return False
        if self.opt.CUR_INDEX >= len(self.opt.LB_LIST):
            self.opt.CUR_INDEX = len(self.opt.LB_LIST) - 1
            self.log('This is the last one.')
            return False
        return True

    def initial_label(self):
        label_path = self.get_label_path()
        with open(label_path, 'r') as f:
            lines = [line.strip().split() for line in f.readlines() if len(line.strip().split()) == 5]
            self.l_pic.labels = list(map(lambda x: self.utils.cxywh2cxyxy([int(x[0]),]+list(map(float,x[1:]))), lines))
        self.log(f'Load label from:\n{label_path}.')
        self.log_length()

    def get_label_path(self):
        lb_path = self.opt.LB_LIST[self.opt.CUR_INDEX]
        return lb_path

    def get_image_path(self):
        lb_name = self.opt.LB_LIST[self.opt.CUR_INDEX].name
        p = self.opt.IMG_DIR / lb_name
        return p.with_suffix('.jpg')

    def find_boxes(self, x, y):
        boxes = []
        for i,cxyxy in enumerate(self.l_pic.labels):
            c, x0, y0, x1, y1 = cxyxy
            if x0 < x and x < x1 and y0 < y and y < y1:
                boxes.append((i,cxyxy))
        return boxes

    def delete_boxes(self, x, y):
        f = 0
        for i, box in self.find_boxes(x, y):
            self.l_pic.labels.remove(box)
            self.log(('Delete [%d' + ' %d' * 4) % tuple(box) + '].')
            f = 1
        if not f: return
        self.log_length()

    def change_boxes(self, x, y):
        tag = int(self.edit_cls.currentText())
        for i, cxyxy in self.find_boxes(x, y):
            self.l_pic.labels[i][0] = tag
            self.log(('Change [%d' + ' %d' * 4) % tuple(self.l_pic.labels[i]) + '].')

    def keyPressEvent(self, e):
        if e.key() == 16777220: # Qt.Key_Enter
            self.func_save()
        elif e.key() == Qt.Key_Backspace:
            self.func_remv()
        elif e.key() == Qt.Key_I:
            self.btn_mode_ist.setChecked(True)
        elif e.key() == Qt.Key_D:
            self.btn_mode_del.setChecked(True)
        elif e.key() == Qt.Key_C:
            self.btn_mode_chg.setChecked(True)
        elif e.key() == Qt.Key_V:
            self.view_state = not self.view_state
            self.update()
        elif e.key() in self.opt.NUM_keys:
            self.edit_cls.setCurrentIndex(e.key()-Qt.Key_0)
        elif e.key() == Qt.Key_Left:
            self.func_prev()
        elif e.key() == Qt.Key_Right:
            self.func_next()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoLabel tool for yolov5')
    parser.add_argument(
        '-m', '--mode', type=str,
        help='set tool mode, either head or half',
        choices=['head', 'half'],
        default='head'
    )
    args = parser.parse_args()
    mode = args.mode
    app = QApplication(sys.argv)
    tool = Eiuyc_Label_Tool(mode)
    sys.exit(app.exec_())
