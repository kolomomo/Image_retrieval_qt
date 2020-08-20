# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   FileName:    MainWindow.py
   Author:      kolomomo
   Date:        2020/8/13
   UpdateData:  2020/8/13:
-------------------------------------------------
   Description:
   
-------------------------------------------------
"""

import sys
import time
import yaml
from keras.models import Model
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from models.load_models import load_irmodels
from utils.grad_camv2 import make_heatmap, show_heatmap
from qt.retrieval_ui import *
from dataset.database import Database
from retrieval.ir_query import *


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.setupUi(self)
        # 初始化
        self.image_size = 128
        self.color_mode = 'grayscale'
        self.pick_layer = 'feature'
        self.model_name = 'No model'
        self.depth = 100
        self.keys = []
        self.F = None
        self.model = None
        # assert cfg is not None, "配置文件为空"

        #
        self.fname = None  # 打开的检索图像路径
        self.fin = []  # 需要检索的图像特征
        self.prob = None  # 预测结果
        self.pre_f = None   # 预处理后的图像
        self.paths = None
        self.show_layer = None
        # 按钮事件
        self.pushButton.clicked.connect(self.loadfile)  # 加载图片
        self.pushButton_2.clicked.connect(self.query)   # 查询
        self.pushButton_3.clicked.connect(self.loadcfg)  # 加载配置文件
        self.pushButton_4.clicked.connect(self.hotmap)  # 热力图
        self.pushButton_5.clicked.connect(self.close)  # 退出
        self.pushButton_6.clicked.connect(self.reset)  # reset

    # 加载解析配置文件
    def loadcfg(self):

        print("加载配置文件")
        try:
            cfg_file, _ = QFileDialog.getOpenFileName(self, '选择配置文件', '../', '*.yaml')
            print(cfg_file)
            with open(cfg_file, encoding='utf-8') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                print(config)

            self.image_size = config['image_size']
            self.color_mode = config['color_mode']
            self.pick_layer = config['pick_layer']
            self.depth = config['retrieval_depth']
            self.model_name = config['model_name']
            self.show_layer = config['show_layer']
            # 特征加载
            self.textBrowser.clear()
            self.textBrowser.insertPlainText("正在加载......")
            QApplication.processEvents()
            experiment_dir = os.path.join(config['root_dir'], config['experiment_name'])
            feature_path = os.path.join(experiment_dir, 'features')
            sample_cache = '{}-{}-{}'.format(self.model_name, self.pick_layer, config['s_name'])
            feature = os.path.join(feature_path, sample_cache)
            print('加载离线特征：', feature)
            # 获取类别
            S_PATH = config['s_path']  # 检索数据集
            save_S_csv = os.path.join(experiment_dir, 'ir_S.csv')
            sdb = Database(data_path=S_PATH, save_path=save_S_csv)
            model_predict = load_irmodels(config)

            self.keys = sdb.get_class()
            self.F = cPickle.load(open(feature, "rb", True))  # np.load('../feature_ex/Features.npy') # feature set
            self.model = Model(inputs=model_predict.input,
                               outputs=[model_predict.get_layer(self.pick_layer).output, model_predict.output])
            # TODO 改变显示方式
            self.btnState()  # 改变模型下拉栏状态
            self.textBrowser.clear()
            self.textBrowser.insertPlainText("加载完成")

        except Exception:
            print("加载错误")
            self.textBrowser.clear()
            self.textBrowser.insertPlainText("加载错误")

    def reset(self):
        self.fname = None  # 打开的检索图像路径
        self.fin = []  # 需要检索的图像特征
        self.prob = None  # 预测结果
        self.pre_f = None
        self.pick_layer = 'feature'  # 提取注意力卷积层
        self.textBrowser.clear()
        #
        self.image_size = 128
        self.color_mode = 'grayscale'
        self.pick_layer = 'feature'
        self.model_name = 'No model'
        self.depth = 100
        self.keys = []
        self.F = None
        self.model = None
        self.show_layer = None
        #
        scene = QGraphicsScene()
        self.graphicsView.setScene(scene)
        self.tableWidget.clear()
        #
        self.textBrowser.insertPlainText("重置完成")

    def prepare(self, d_img):

        image_size = self.image_size
        color_mode = self.color_mode
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
        if self.fname is None:
            print("选择检索图像")
            self.textBrowser.clear()
            self.textBrowser.insertPlainText("选择检索图像")
            return
        if self.model is None:
            print("选择检索配置文件")
            self.textBrowser.clear()
            self.textBrowser.insertPlainText("选择检索配置文件")
            return
        print("开始检索")
        self.textBrowser.clear()
        self.textBrowser.insertPlainText("开始检索")
        self.pre_f = self.prepare(self.fname)
        fin, self.prob = self.model.predict(self.pre_f)

        fin = np.sum(fin, axis=0)
        fin /= np.sum(fin)  # normalize
        aaa = np.argsort(-self.prob) # 预测的类别
        d = self.depth
        keys = list(self.keys)
        ta = time.time()
        r = single_query_infer(fin, s_samples=self.F, depth=d, d_type=self.comboBox_3.currentIndex())
        tb = time.time()
        fw = '检索深度：{} \r\n检索耗时：{:.4}秒 \r\n预测类别1:{},置信度：{:.2%} \r\n预测类别2:{},置信度：{:.2%}'.format(d, (tb - ta),
                                                                                              keys[aaa[0][0]],
                                                                                              self.prob[0][aaa[0][0]],
                                                                                              keys[aaa[0][1]],
                                                                                              self.prob[0][aaa[0][1]])
        print(fw)
        self.viewfiles(r, fw)

    def loadfile(self):
        print("加载图片")
        try:
            self.fname, _ = QFileDialog.getOpenFileName(self, '选择图片', '../',
                                                        'Image files(*.jpg *.gif *.png)')
            img = QImage()
            img.load(self.fname)
            # 调整图像大小与graphicsView一致
            img = img.scaled(self.graphicsView.width(), self.graphicsView.height())
            scene = QGraphicsScene()
            scene.addPixmap(QPixmap().fromImage(img))
            self.textBrowser.clear()
            self.textBrowser.insertPlainText(str(self.fname))
            self.graphicsView.setScene(scene)
        except Exception:
            print("加载错误")

    def btnState(self):
        # self.comboBox.currentIndex() == 1:
        self.comboBox.setItemText(0, self.model_name)

    def viewfiles(self, r, fw):
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
            # print('%s i=%d  j=%d' % (r[k], i, j))
            # 将条目加载到相应行列中
            self.tableWidget.setItem(i, j, item)

    # TODO 优化
    def hotmap(self):
        try:
            if self.show_layer is None:
                self.textBrowser.clear()
                self.textBrowser.insertPlainText("未指定可视化卷积层")
                return
            heatmap = make_heatmap(self.pre_f, self.model, self.prob, layer=self.show_layer)
            self.textBrowser.clear()
            self.textBrowser.insertPlainText("按ESC退出")
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
