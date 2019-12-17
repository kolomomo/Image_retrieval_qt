# import os
# #
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import time
import cv2
import numpy as np
import keras

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from keras.models import load_model
from vis.grad_camv2 import make_heatmap, show_heatmap

from pyqt5.myretrieval import *
from retrieval.ir_query_v2 import *
from models.my_models import CenterVLAD
from keras.backend import l2_normalize, expand_dims

from keras.backend import clear_session

keys = ['1121-110-213-700', '1121-110-411-700', '1121-110-414-700', '1121-110-415-700', '1121-115-700-400',
        '1121-115-710-400', '1121-116-917-700', '1121-120-200-700', '1121-120-310-700', '1121-120-311-700',
        '1121-120-320-700', '1121-120-330-700', '1121-120-331-700', '1121-120-413-700', '1121-120-421-700',
        '1121-120-422-700', '1121-120-433-700', '1121-120-434-700', '1121-120-437-700', '1121-120-438-700',
        '1121-120-441-700', '1121-120-442-700', '1121-120-451-700', '1121-120-452-700', '1121-120-454-700',
        '1121-120-462-700', '1121-120-463-700', '1121-120-514-700', '1121-120-515-700', '1121-120-516-700',
        '1121-120-517-700', '1121-120-800-700', '1121-120-911-700', '1121-120-914-700', '1121-120-915-700',
        '1121-120-918-700', '1121-120-919-700', '1121-120-91a-700', '1121-120-921-700', '1121-120-922-700',
        '1121-120-930-700', '1121-120-933-700', '1121-120-934-700', '1121-120-942-700', '1121-120-943-700',
        '1121-120-950-700', '1121-120-951-700', '1121-120-956-700', '1121-120-961-700', '1121-120-962-700',
        '1121-127-700-400', '1121-127-700-500', '1121-129-700-400', '1121-12f-466-700', '1121-12f-467-700',
        '1121-200-411-700', '1121-210-213-700', '1121-210-230-700', '1121-210-310-700', '1121-210-320-700',
        '1121-210-330-700', '1121-210-331-700', '1121-220-213-700', '1121-220-230-700', '1121-220-310-700',
        '1121-220-330-700', '1121-228-310-700', '1121-229-310-700', '1121-230-462-700', '1121-230-463-700',
        '1121-230-911-700', '1121-230-914-700', '1121-230-915-700', '1121-230-921-700', '1121-230-922-700',
        '1121-230-930-700', '1121-230-934-700', '1121-230-942-700', '1121-230-943-700', '1121-230-950-700',
        '1121-230-953-700', '1121-230-961-700', '1121-230-962-700', '1121-240-413-700', '1121-240-421-700',
        '1121-240-422-700', '1121-240-433-700', '1121-240-434-700', '1121-240-437-700', '1121-240-438-700',
        '1121-240-441-700', '1121-240-442-700', '1121-320-941-700', '1121-420-212-700', '1121-420-213-700',
        '1121-430-213-700', '1121-430-215-700', '1121-460-216-700', '1121-490-310-700', '1121-490-415-700',
        '1121-490-915-700', '1121-4a0-310-700', '1121-4a0-414-700', '1121-4a0-914-700', '1121-4a0-918-700',
        '1121-4b0-233-700', '1122-220-333-700', '1123-110-500-000', '1123-112-500-000', '1123-121-500-000',
        '1123-127-500-000', '1123-211-500-000', '1124-310-610-625', '1124-310-620-625', '1124-410-610-625',
        '1124-410-620-625']

M_PATH = '/home/wbo/PycharmProjects/Image_retrieval_qt/pyqt5/query_models/query_CenterVLAD_Net_X.h5'
F_PATH = '/home/wbo/PycharmProjects/Image_retrieval_qt/pyqt5/qurey_samples/CenterVLAD_Net_X-feature-SData'




class LoadThread(QThread):
    sinOut = pyqtSignal(list, str)

    def __init__(self):
        super().__init__()

    def set_value(self, fname, dtype, model, F):
        self.fname = fname
        self.dtype = dtype
        self.model = model
        self.F = F

    def set_input(self, fname):
        self.fname=fname



    def run(self):
        print("开始检索")
        fin, prob = self.model.predict(self.prepare(self.fname))
        fin = np.sum(fin, axis=0)
        fin /= np.sum(fin)  # normalize
        aaa = np.argsort(-prob)
        d = 50
        ta = time.time()
        r = single_query_infer(fin, s_samples=self.F, depth=d, d_type=self.dtype)
        tb = time.time()
        fw = '检索深度：{} \r\n检索耗时：{:.4}秒 \r\n预测类别1:{},置信度：{:.2%} \r\n预测类别2:{},置信度：{:.2%}'.format(d, (tb - ta),
                                                                                              keys[aaa[0][0]],
                                                                                              prob[0][aaa[0][0]],
                                                                                              keys[aaa[0][1]],
                                                                                              prob[0][aaa[0][1]])

        self.sinOut.emit(r, fw)
        # 销毁当前TF图并创建一个新的TF图，避免旧模型/图层的混乱。
        # clear_session()
        self.sleep(2)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.fname = None  # 打开的检索图像路径
        self.fin = []  # 需要检索的图像特征
        self.prob = None  # 预测结果
        self.pre_f = None
        self.pick_layer = 'multiply_6'  # 提取注意力卷积层
        self.F = cPickle.load(open(F_PATH, "rb", True))  # np.load('../feature_ex/Features.npy') # feature set
        self.paths = None  # np.load('../feature_ex/img_paths.npy')# img paths set
        self.model = load_model(M_PATH,
                                    custom_objects={'CenterVLAD': CenterVLAD, 'l2_normalize': l2_normalize,
                                               'expand_dims': expand_dims})
        print(self.model.summary())
        # load_model('../feature_ex/my_model.model')# model for retrieval

        # 按钮事件
        self.pushButton.clicked.connect(self.loadfile)  # 加载图片
        self.pushButton_2.clicked.connect(self.query)
        self.pushButton_3.clicked.connect(self.loadset)  # 加载检索库文件
        self.pushButton_4.clicked.connect(self.hotmap)  # 热力图
        self.pushButton_5.clicked.connect(self.close)  # 退出
        self.pushButton_6.clicked.connect(self.reset)  # reset

        #
        self.comboBox.currentIndexChanged.connect(self.btnState)

        self.loadThread = LoadThread()
        self.loadThread.sinOut.connect(self.viewfiles)

    def reset(self):
        self.fname = None  # 打开的检索图像路径
        self.fin = []  # 需要检索的图像特征
        self.prob = None  # 预测结果
        self.pre_f = None
        self.pick_layer = 'multiply_6'  # 提取注意力卷积层
        self.textBrowser.clear()
        self.textBrowser.insertPlainText("重置完成")

    def prepare(self, d_img):
        image_size = 128
        color_mode = 'grayscale'
        try:
            img = image.load_img(d_img, target_size=(image_size, image_size), color_mode=color_mode)
            img = image.img_to_array(img)
            img = img / 255.
            img = np.expand_dims(img, axis=0)
            self.textBrowser.clear()
            self.textBrowser.insertPlainText("图像处理完成")
            print("图像预处理完成")
            return img
        except Exception:
            self.textBrowser.clear()
            self.textBrowser.insertPlainText("图像处理错误")
            print("图像预处理错误")

    def query(self):
        # get input feature
        assert self.fname is not None
        print("开始检索")
        self.textBrowser.clear()
        self.textBrowser.insertPlainText("开始检索")
        self.pre_f = self.prepare(self.fname)
        fin, self.prob = self.model.predict(self.pre_f)
        fin = np.sum(fin, axis=0)
        fin /= np.sum(fin)  # normalize
        aaa = np.argsort(-self.prob)
        d = 100
        ta = time.time()
        r = single_query_infer(fin, s_samples=self.F, depth=d, d_type=self.comboBox_3.currentIndex())
        tb = time.time()
        fw = '检索深度：{} \r\n检索耗时：{:.4}秒 \r\n预测类别1:{},置信度：{:.2%} \r\n预测类别2:{},置信度：{:.2%}'.format(d, (tb - ta),
                                                                                              keys[aaa[0][0]],
                                                                                              self.prob[0][aaa[0][0]],
                                                                                              keys[aaa[0][1]],
                                                                                              self.prob[0][aaa[0][1]])

        self.viewfiles(r, fw)

    def loadfile(self):
        print("load_file")
        try:
            self.fname, _ = QFileDialog.getOpenFileName(self, '选择图片', '/home/w/PycharmProjects/SEA/datasets/ir_val_Q',
                                                        'Image files(*.jpg *.gif *.png)')
            img = QImage()
            img.load(self.fname)
            # 调整图像大小与graphicsView一致
            img = img.scaled(self.graphicsView.width(), self.graphicsView.height())
            scene = QGraphicsScene()
            scene.addPixmap(QPixmap().fromImage(img))
            self.graphicsView.setScene(scene)
        except Exception:
            print("load error")

    def loadset(self):
        print("load_set")
        try:
            fname, _ = QFileDialog.getOpenFileName(self, '选择图片', 'c:\\', 'Image files(*.npy *.npz, *)')
            print(fname)
            self.F = cPickle.load(open(fname, "rb", True))
            print("load success")
            print(self.F)
        except Exception:
            print("load error")

    def btnState(self):

        self.textBrowser.clear()
        self.textBrowser.insertPlainText("正在加载模型......")
        QApplication.processEvents()
        if self.comboBox.currentIndex() == 1:
            self.model = load_model('/home/wbo/PycharmProjects/Image_retrieval_qt/pyqt5/query_models/query_SECA.h5')
            self.textBrowser.clear()
            self.textBrowser.insertPlainText("正在加载检索库......")
            QApplication.processEvents()
            time.sleep(1)
            self.F = cPickle.load(open('/home/wbo/PycharmProjects/Image_retrieval_qt/pyqt5/qurey_samples/SECA_X-feature-SData', "rb", True))
            self.pick_layer = 'multiply_6'

        elif self.comboBox.currentIndex() == 0:
            self.textBrowser.clear()
            self.textBrowser.insertPlainText("正在加载模型......")
            QApplication.processEvents()
            self.model = load_model('/home/wbo/PycharmProjects/Image_retrieval_qt/pyqt5/query_models/query_CenterVLAD_Net_X.h5',
                                    custom_objects={'CenterVLAD': CenterVLAD, 'l2_normalize': l2_normalize,
                                                    'expand_dims': expand_dims})
            self.textBrowser.clear()
            self.textBrowser.insertPlainText("正在加载检索库......")
            QApplication.processEvents()
            time.sleep(1)
            self.F = cPickle.load(open('/home/wbo/PycharmProjects/Image_retrieval_qt/pyqt5/qurey_samples/CenterVLAD_Net_X-feature-SData', "rb", True))
            self.pick_layer = 'multiply_6'

        self.textBrowser.clear()
        self.textBrowser.insertPlainText("加载完成")

    def viewfiles(self, r, fw):
        '''
        检索图像显示/分类结果显示
        '''

        self.textBrowser.clear()
        self.textBrowser.insertPlainText("开始检索....")
        QApplication.processEvents()
        print("开始检索")
        self.textBrowser.moveCursor(self.textBrowser.textCursor().End)
        self.textBrowser.clear()
        self.textBrowser.insertPlainText(fw)
        depth = len(r)
        for i in range(6):  # 列宽150
            self.tableWidget.setColumnWidth(i, 150)
        for i in range(20):  # 行高150
            self.tableWidget.setRowHeight(i, 150)

        for k in range(depth):
            i = k / 6
            j = k % 6
            # 实例化表格窗口条目
            item = QTableWidgetItem()
            # 用户点击表格时，图片被选中
            item.setFlags(Qt.ItemIsEnabled)
            # 图片路径设置与图片加载
            icon = QIcon(r[k])
            item.setIcon(QIcon(icon))
            # 输出当前进行的条目序号
            print('%s i=%d  j=%d' % (r[k], i, j))
            # 将条目加载到相应行列中
            self.tableWidget.setItem(i, j, item)

    def hotmap(self):
        try:
            heatmap = make_heatmap(self.pre_f, self.model, self.prob, layer=self.pick_layer)
            self.textBrowser.clear()
            self.textBrowser.insertPlainText("按ESC退出热力图！")
            show_heatmap(self.fname, heatmap)
            self.textBrowser.clear()
        except:
            self.textBrowser.clear()
            self.textBrowser.insertPlainText("打开热力图失败，未知错误....")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec_())