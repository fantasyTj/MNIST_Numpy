import sys
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image, ImageQt

from final_gui_package.final_gui import Ui_MainWindow
from final_gui_package.final_paintboard import PaintBoard

from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QApplication
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import QSize

# 自适应窗口大小
from PyQt5 import QtCore
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

from deepcovnet import DeepConvNet
from components.img_split import split

MODE_MNIST = 1  # MNIST随机抽取
MODE_WRITE = 2  # 手写输入

# 读取MNIST数据集
(_, _), (x_test, _) = load_mnist(normalize=True, flatten=False, one_hot_label=True)

# 初始化网络
network = DeepConvNet()
network.load_params("deepconvnet_params.pkl")

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # 初始化参数
        self.mode = MODE_MNIST
        self.result = [0]

        # 初始化UI
        self.setupUi(self)
        self.center()

        # 初始化画板
        self.paintBoard = PaintBoard(self, Size=QSize(691, 211), Fill=QColor(0, 0, 0, 0))
        self.paintBoard.setPenColor(QColor(0, 0, 0, 0))
        self.verticalLayout.addWidget(self.paintBoard)
        self.paintBoard.setBoardFill(QColor(0, 0, 0, 255))
        # self.paintBoard.setPenColor(QColor(255, 255, 255, 255))

        self.clearDataArea()

    # 窗口居中
    def center(self):
        # 获得窗口
        framePos = self.frameGeometry()
        # 获得屏幕中心点
        scPos = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        framePos.moveCenter(scPos)
        self.move(framePos.topLeft())

    # 窗口关闭事件
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Attention',
                                     "Sure to quit?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.Yes)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

            # 清除数据待输入区

    def clearDataArea(self):
        self.paintBoard.Clear()
        self.MnistShow.clear()
        self.Result.clear()
        self.result = [0]

    """
    回调函数
    """

    # 模式下拉列表回调
    def cbBox_Mode_Callback(self, text):
        if text == "1.Mnist数据集":
            self.mode = MODE_MNIST
            self.clearDataArea()
            self.pbtGetMnist.setEnabled(True)

            # self.paintBoard.setBoardFill(QColor(0, 0, 0, 0))
            self.paintBoard.setPenColor(QColor(0, 0, 0, 0))

        elif text == "2.手写数字":
            self.mode = MODE_WRITE
            self.clearDataArea()
            self.pbtGetMnist.setEnabled(False)

            # 更改背景
            self.paintBoard.setBoardFill(QColor(0, 0, 0, 255))
            self.paintBoard.setPenColor(QColor(255, 255, 255, 255))

    # 数据清除
    def pbtClear_Callback(self):
        self.clearDataArea()

    # 识别
    def pbtPredict_Callback(self):
        __img, img_array = [], []  # 将图像统一从qimage->pil image -> np.array [1, 1, 28, 28]

        # 获取qimage格式图像
        if self.mode == MODE_MNIST:
            __img = self.MnistShow.pixmap()  # label内若无图像返回None
            if __img == None:  # 无图像则用纯黑代替
                # __img = QImage(224, 224, QImage.Format_Grayscale8)
                __img = ImageQt.ImageQt(Image.fromarray(np.uint8(np.zeros([221, 221]))))
            else:
                __img = __img.toImage()
        elif self.mode == MODE_WRITE:
            __img = self.paintBoard.getContentAsQImage()

        # 转换成pil image类型处理
        pil_img = ImageQt.fromqimage(__img)
        if self.mode==MODE_MNIST:
            pil_img = pil_img.resize((28, 28),Image.ANTIALIAS)
            img_array=np.array(pil_img.convert('L')).reshape((1,1,28,28)) / 255.0
            result = np.argmax(network.predict(img_array))
        elif self.mode==MODE_WRITE:
            # pil_img = pil_img.resize((91, 28), Image.ANTIALIAS)
            img_array=np.array(pil_img.convert('L'))
            img_array=split(img_array)
            result = np.argmax(network.predict(img_array.reshape(img_array.shape[0],1,28,28)).reshape(-1,10), axis=1)
            result=np.asarray(result,dtype=str)
            result=int(''.join(result))
            pass

        self.Result.setText("%d" % result)

    # 随机抽取
    def pbtGetMnist_Callback(self):
        self.clearDataArea()

        # 随机抽取一张测试集图片，放大后显示
        img = x_test[np.random.randint(0, 9999)]  # shape:[1,28,28]
        img = img.reshape(28, 28)  # shape:[28,28]

        img = img * 0xff  # 恢复灰度值大小
        pil_img = Image.fromarray(np.uint8(img))
        pil_img = pil_img.resize((211, 211))  # 图像放大显示

        # 将pil图像转换成qimage类型
        qimage = ImageQt.ImageQt(pil_img)

        # 将qimage类型图像显示在label
        pix = QPixmap.fromImage(qimage)
        self.MnistShow.setPixmap(pix)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Gui = MainWindow()
    Gui.show()
    sys.exit(app.exec_())