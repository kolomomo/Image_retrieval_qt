# -*- coding: utf-8 -*-

"""
Module implementing Dialog.
"""

from PyQt5.QtCore import *
from PyQt5.QtWidgets import QDialog, QMessageBox

from PyQt5 import QtWidgets
from Ui_login import Ui_Dialog
import webbrowser, pymysql


class Dialog(QDialog, Ui_Dialog):
    """
    Class documentation goes here.
    """
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='liaojun', db='pyqt', charset='utf8', )
    cursor = conn.cursor()
    # 第二步：连接数据库pyqt_clicked1是一个信号连接
    pyqt_clicked1 = pyqtSignal()

    def __init__(self, parent=None):
        """
        Constructor
        @param parent reference to the parent widget
        @type QWidget
        """
        super(Dialog, self).__init__(parent)
        self.setupUi(self)
        # 第三步连接数据库pyqt_clicked1是一个信号连接
        self.pyqt_clicked1.connect(self.on_pushButton_login_clicked)

    @pyqtSlot()
    def on_pushButton_clicked(self):
        """
        Slot documentation goes here.
        """
        # 打开官网找回密码
        mystr1 = self.lineEdit.text()
        url = 'https://aq.qq.com/v2/uv_aq/html/reset_pwd/pc_reset_pwd_input_account.html?v=3.0&old_ver_account='
        url += mystr1
        webbrowser.open(url)

    @pyqtSlot()
    def on_pushButton_2_clicked(self):
        """
        Slot documentation goes here.
        """
        # 打开官网去注册
        webbrowser.open('ssl.zc.qq.com')

    @pyqtSlot()
    def on_pushButton_login_clicked(self):
        """
        Slot documentation goes here.
        """
        # 查询用户
        # 设置查询语句的模型
        self.sqlstring = "select * from students where "
        temp_sqlstring = self.sqlstring
        mystr1 = self.lineEdit.text()
        mystr2 = self.lineEdit_2.text()
        if mystr1 != '':
            if mystr2 != '':
                temp_sqlstring += "sname = '" + self.lineEdit.text() + "'"
                print(temp_sqlstring)
                if 1 == 1:
                    # 查询数据库
                    self.cursor.execute(temp_sqlstring)
                    # 接受查询结果
                    data = self.cursor.fetchone()
                    if data != None:
                        # 假设data[0]中存储的是密码，实际上我的数据库中存的是sid
                        if mystr2 == data[0]:
                            my_botton2 = QMessageBox.information(self, 'information', '登录成功')
                            # 调用主窗口
                            from main_index import Dialog
                            d = Dialog()
                            d.exec_()
                            d.hide()
                        else:
                            my_botton2 = QMessageBox.warning(self, 'waring', '密码错误')
                    else:
                        my_botton2 = QMessageBox.warning(self, 'waring', '用户不存在')
                else:
                    my_botton2 = QMessageBox.warning(self, 'waring', '登录错误')
            else:
                my_botton2 = QMessageBox.warning(self, 'waring', '密码为空')
        else:
            my_botton2 = QMessageBox.warning(self, 'waring', '用户名为空')

    @pyqtSlot()
    def on_pushButton_saoma_clicked(self):
        """
        Slot documentation goes here.
        """
        # 调用子扫码窗口进行扫码
        from saoma import Dialog
        d = Dialog()
        d.exec_()