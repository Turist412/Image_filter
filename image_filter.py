from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import traceback
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets

def fact(n):
    a = 1
    for x in range(1, n + 1):
        a *= n
    return a

def C(n, i):
    return fact(n) // fact(i) // fact(n - i)

def B(n, i, t):
    return C(n, i) * t ** i * (1 - t) ** (n - i)

class Field(QWidget):
    def __init__(self, parent, points=[[0,0],[100,100],[200,200],[255,0]], field_size=(256, 256), cell_size=2, point_size=3):
        super(Field, self).__init__(parent)
        self.parent = parent
        self.field_size = field_size
        self.field_width = field_size[0]
        self.field_height = field_size[1]
        self.gamma_dict = {x: x for x in range(256)}
        self.cell_size = cell_size
        self.point_size = point_size

        self.window_size = (field_size[0] * cell_size, field_size[1] * cell_size)
        self.window_width = self.window_size[0]
        self.window_height = self.window_size[1]
        self.current_point = 0
        self.condition = np.zeros(self.field_size)
        self.history = [self.condition]
        timer = QTimer(self)
        timer.timeout.connect(self.update)
        timer.start(1000 // 30)
        self.setGeometry(0, 0, self.window_width, self.window_height)
        self.setStyleSheet("background : black;")
        self.last_cur_pos = (0, 0)
        self.points = points
        for point in self.points:
            for x in range(point[0] - self.point_size, point[0] + self.point_size + 1):
                for y in range(point[1] - self.point_size, point[1] + self.point_size + 1):
                    if 0 <= x < 256 and 0 <= y < 256 and (point[0] - x) ** 2 + (point[1] - y) ** 2 <= self.point_size ** 2:
                        self.condition[x, y] = 1
        self.show()

    def load(self, array):
        arr_size = array.shape[0]
        if arr_size > self.condition.shape[0] or arr_size not in (50, 100, 200, 400):
            raise Exception("wrong_array_size")
        self.condition = np.zeros(self.field_size)
        self.condition[(self.field_width - arr_size) // 2:arr_size, (self.field_width - arr_size) // 2:arr_size] = array
        self.history = [self.condition]

    def clear(self):
        self.condition = np.zeros(self.field_size)
        self.history = [self.condition]

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.drawPoints(qp)
        qp.end()

    def mousePressEvent(self, e):
        x = min(max(2*e.pos().x() // self.cell_size, 0), 255)
        y = min(max(2*e.pos().y() // self.cell_size, 0), 255)
        cur_pos = [x, y]
        self.points[self.current_point + 1][0] = cur_pos[0]
        self.points[self.current_point + 1][1] = cur_pos[1]
        self.current_point = not self.current_point



    def drawPoints(self, qp):
        qp.setBrush(Qt.black)
        qp.drawRect(0, 0, self.window_size[0], self.window_size[1])
        qp.setPen(Qt.white)
        size = self.size()
        self.condition *= 0
        self.bezier_curve()
        for point in self.points:
            for x in range(point[0] - self.point_size, point[0] + self.point_size + 1):
                for y in range(point[1] - self.point_size, point[1] + self.point_size + 1):
                    if 0 <= x < 256 and 0 <= y < 256 and abs(point[0] - x) <= self.point_size and \
                            abs(point[1] - y) <= self.point_size:
                        self.condition[x, y] = 1
        for x in range(self.field_width):
            for y in range(self.field_height):
                if self.condition[x, y] == 1:
                    qp.drawPoint(x, y)

    def bezier_curve(self):
        try:
            T = [t * 0.001 for t in range(0, int(1 / 0.001) + 1)]
            for t in T:
                X = (1 - t) ** 3 * self.points[0][0] + 3 * (1 - t) ** 2 * t * self.points[1][0] \
                    + 3 * (1 - t) * t ** 2 * self.points[2][0] + t ** 3 * self.points[3][0]
                Y = (1 - t) ** 3 * self.points[0][1] + 3 * (1 - t) ** 2 * t * self.points[1][1] \
                    + 3 * (1 - t) * t ** 2 * self.points[2][1] + t ** 3 * self.points[3][1]
                self.condition[min(int(X), 255), min(int(Y), 255)] = 1
        except Exception as e:
            print(traceback.print_exc())


class Ui_Image_filter(object):
    def setupUi(self, Image_filter):
        self.image = None
        self.temp = None
        self.bezier_points = [[0,255],[100,100],[200,200],[255,0]]
        self.result_image = None
        self.temp_result = None
        self.future_filter = np.array([[1, 2, 1],[2, 4, 2], [1, 2, 1]]) * (1/16)
        self.filters = [np.array([[1, 2, 1],[2, 4, 2], [1, 2, 1]]) * (1/16),
                        np.array([[-1, -2, -1],[-2, 22, -2],[-1, -2, -1]]) * (1/10),
                        np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]]),
                        np.array([[1, 0, 0],[0, 0, 0],[0, 0, -1]]),
                        np.array([[-2, -1, 0],[-1, 1, 1],[0, 1, 2]]),
                        np.array([[-1, -1, -1],[-1, 9, 1],[-1, -1, -1]]) * (1/9)]
        Image_filter.setObjectName("Image_filter")
        Image_filter.resize(1013, 811)
        self.centralwidget = QtWidgets.QWidget(Image_filter)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 70, 491, 401))
        self.label.setStyleSheet("background-color: white;")
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(170, 30, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(570, 30, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(840, 30, 93, 28))
        self.pushButton_3.setObjectName("pushButton_3")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(320, 30, 150, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.field = Field(None, points = self.bezier_points)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 490, 255, 255))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout.addWidget(self.field)
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(330, 480, 171, 28))
        self.pushButton_4.setObjectName("pushButton_4")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(510, 70, 491, 401))
        self.label_2.setStyleSheet("background-color: white;")
        self.label_2.setObjectName("label_2")
        Image_filter.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Image_filter)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1013, 26))
        self.menubar.setObjectName("menubar")
        Image_filter.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Image_filter)
        self.statusbar.setObjectName("statusbar")
        Image_filter.setStatusBar(self.statusbar)

        self.retranslateUi(Image_filter)
        self.pushButton.clicked.connect(self.add_image)
        self.pushButton_2.clicked.connect(self.apply_filter)
        self.pushButton_3.clicked.connect(self.save_result)
        self.comboBox.activated.connect(self.filter_pick)
        self.pushButton_4.clicked.connect(self.gamma_correction)
        QtCore.QMetaObject.connectSlotsByName(Image_filter)

    def gamma_correction(self):
        try:
            self.bezier_curve()

            mp = np.arange(0, 256)
            mp[list(self.gamma_dict.keys())] = list(self.gamma_dict.values())
            self.result_image2 = mp[self.image]
            cv2.imwrite("tmp.bmp", self.result_image2)
            pixmap = QPixmap("tmp.bmp")

            heighti, widthi = pixmap.height(), pixmap.width()

            if heighti < widthi:
                width = self.label.width()
                pixmap = pixmap.scaledToWidth(width)
            else:
                height = self.label.height()
                pixmap = pixmap.scaledToHeight(height)
            self.label_2.setPixmap(pixmap)
        except Exception as e:
            print(list(self.gamma_dict.keys()))
            print(traceback.print_exc())

    def bezier_curve(self):
        T = [t * 0.001 for t in range(0, int(1 / 0.001) + 1)]
        self.gamma_dict = {x: [] for x in range(256)}
        for t in T:
            X = (1 - t) ** 3 * self.bezier_points[0][0] + 3 * (1 - t) ** 2 * t * self.bezier_points[1][0] \
                + 3 * (1 - t) * t ** 2 * self.bezier_points[2][0] + t ** 3 * self.bezier_points[3][0]
            Y = (1 - t) ** 3 * self.bezier_points[0][1] + 3 * (1 - t) ** 2 * t * self.bezier_points[1][1] \
                + 3 * (1 - t) * t ** 2 * self.bezier_points[2][1] + t ** 3 * self.bezier_points[3][1]
            X = int(X)
            Y = int(255 - Y)
            self.gamma_dict[X].append(Y)
        for x in range(256):
            self.gamma_dict[x] = int(np.mean(self.gamma_dict[x]))

    def save_result(self):
        try:
            fname = QFileDialog.getSaveFileName(None, 'Save', 'result.png')
            cv2.imwrite(fname[0], self.temp_result)
        except:
            pass

    def filter_pick(self, filter):
        self.future_filter = self.filters[filter]

    def add_image(self):
        file, _ = QFileDialog.getOpenFileName(None, 'Open File', './', "Image (*.png *.jpg *jpeg)")
        if file:
            pixmap = QPixmap(file)

            self.image = cv2.imread(file)
            self.temp = self.image.copy()
            heighti, widthi = pixmap.height(), pixmap.width()

            if heighti < widthi:
                width = self.label.width()
                pixmap = pixmap.scaledToWidth(width)
            else:
                height = self.label.height()
                pixmap = pixmap.scaledToHeight(height)
            self.label.setPixmap(pixmap)

    def apply_filter(self):

        self.result_image = np.zeros_like(self.image)
        self.result_image = cv2.filter2D(src=self.image, ddepth=-1, kernel=self.future_filter)
        if np.array_equal(self.future_filter , np.array([[1, 0, 0],[0, 0, 0],[0, 0, -1]])):
            self.result_image += 128
            self.result_image[self.result_image > 255] = 255
            self.result_image[self.result_image < 0] = 0
        self.temp_result = self.result_image.copy()
        cv2.imwrite("tmp.bmp", self.result_image)
        pixmap = QPixmap("tmp.bmp")

        heighti, widthi = pixmap.height(), pixmap.width()

        if heighti < widthi:
            width = self.label.width()
            pixmap = pixmap.scaledToWidth(width)
        else:
            height = self.label.height()
            pixmap = pixmap.scaledToHeight(height)
        self.label_2.setPixmap(pixmap)

    def retranslateUi(self, Image_filter):
        _translate = QtCore.QCoreApplication.translate
        Image_filter.setWindowTitle(_translate("Image_filter", "MainWindow"))
        self.pushButton.setText(_translate("Image_filter", "Load image"))
        self.pushButton_2.setText(_translate("Image_filter", "Result"))
        self.pushButton_3.setText(_translate("Image_filter", "Save result"))
        self.comboBox.setItemText(0, _translate("Image_filter", "Blur"))
        self.comboBox.setItemText(1, _translate("Image_filter", "Quality up"))
        self.comboBox.setItemText(2, _translate("Image_filter", "Find borders"))
        self.comboBox.setItemText(3, _translate("Image_filter", "Imprint"))
        self.comboBox.setItemText(4, _translate("Image_filter", "Retro"))
        self.comboBox.setItemText(5, _translate("Image_filter", "Reduce brightness"))
        self.pushButton_4.setText(_translate("Image_filter", "Apply brightness correction"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Image_filter = QtWidgets.QMainWindow()
    ui = Ui_Image_filter()
    ui.setupUi(Image_filter)
    Image_filter.show()
    sys.exit(app.exec_())
