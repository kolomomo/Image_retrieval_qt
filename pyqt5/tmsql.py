import sys
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import pymysql

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):

        # self.conn = pymysql.connect(host='152.136.109.167', port=3306, user='wb', password='wewe1011', db='person',charset='utf8',)

        # self.cur = self.conn.cursor()

        self.sqlstring = "select * from students where "
        MainWindow.setObjectName("MainWindow")

        MainWindow.resize(760, 440)

        # 根据窗口的大小固定大小 这里相当于设置全屏
        MainWindow.setFixedSize(MainWindow.width(), MainWindow.height())

        self.centralwidget =  QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame =  QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(10, 10, 491, 121))
        self.frame.setFrameShape( QFrame.StyledPanel)
        self.frame.setFrameShadow( QFrame.Raised)
        self.frame.setObjectName("frame")
        self.check_Sid =  QCheckBox(self.frame)
        self.check_Sid.setGeometry(QtCore.QRect(20, 10, 71, 16))
        self.check_Sid.setObjectName("check_Sid")
        self.check_Sage =  QCheckBox(self.frame)
        self.check_Sage.setGeometry(QtCore.QRect(20, 70, 71, 16))
        self.check_Sage.setObjectName("check_Sage")
        self.check_Sname =  QCheckBox(self.frame)
        self.check_Sname.setGeometry(QtCore.QRect(20, 40, 71, 16))
        self.check_Sname.setObjectName("check_Sname")
        self.check_Ssex =  QCheckBox(self.frame)
        self.check_Ssex.setGeometry(QtCore.QRect(20, 100, 71, 16))
        self.check_Ssex.setObjectName("check_Ssex")
        self.Sid =  QLineEdit(self.frame)
        self.Sid.setGeometry(QtCore.QRect(90, 10, 113, 16))
        self.Sid.setObjectName("Sid")
        self.Sname =  QLineEdit(self.frame)
        self.Sname.setGeometry(QtCore.QRect(90, 40, 113, 16))
        self.Sname.setObjectName("Sname")
        self.first_Sage =  QLineEdit(self.frame)
        self.first_Sage.setGeometry(QtCore.QRect(90, 70, 41, 16))
        self.first_Sage.setObjectName("first_Sage")
        self.Ssex =  QLineEdit(self.frame)
        self.Ssex.setGeometry(QtCore.QRect(90, 100, 113, 16))
        self.Ssex.setObjectName("Ssex")
        self.label =  QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(140, 70, 16, 16))
        self.label.setObjectName("label")
        self.last_Sage =  QLineEdit(self.frame)
        self.last_Sage.setGeometry(QtCore.QRect(160, 70, 41, 16))
        self.last_Sage.setObjectName("last_Sage")
        self.check_Sdept =  QCheckBox(self.frame)
        self.check_Sdept.setGeometry(QtCore.QRect(270, 40, 71, 16))
        self.check_Sdept.setObjectName("check_Sdept")
        self.Sdept =  QLineEdit(self.frame)
        self.Sdept.setGeometry(QtCore.QRect(340, 40, 113, 16))
        self.Sdept.setObjectName("Sdept")
        self.Sclass =  QLineEdit(self.frame)
        self.Sclass.setGeometry(QtCore.QRect(340, 10, 113, 16))
        self.Sclass.setObjectName("Sclass")
        self.check_Sclass =  QCheckBox(self.frame)
        self.check_Sclass.setGeometry(QtCore.QRect(270, 10, 71, 16))
        self.check_Sclass.setObjectName("check_Sclass")
        self.Saddr =  QLineEdit(self.frame)
        self.Saddr.setGeometry(QtCore.QRect(340, 70, 113, 16))
        self.Saddr.setObjectName("Saddr")
        self.check_Saddr =  QCheckBox(self.frame)
        self.check_Saddr.setGeometry(QtCore.QRect(270, 70, 71, 16))
        self.check_Saddr.setObjectName("check_Saddr")
        self.find =  QPushButton(self.frame)
        self.find.setGeometry(QtCore.QRect(380, 100, 75, 21))
        self.find.setObjectName("find")
        self.find.clicked.connect(self.find_btn)
        self.sql_out =  QTextBrowser(self.centralwidget)
        self.sql_out.setGeometry(QtCore.QRect(10, 140, 740, 61))
        self.sql_out.setObjectName("sql_out")
        self.result_out =  QTableWidget(self.centralwidget)
        self.result_out.setEditTriggers( QAbstractItemView.NoEditTriggers)  # 不可编辑表格
        self.result_out.setGeometry(QtCore.QRect(10, 210, 740, 171))
        self.result_out.setObjectName("result_out")
        self.result_out.setColumnCount(7)
        self.result_out.setRowCount(10)
        self.result_out.resizeColumnsToContents()
        self.result_out.resizeRowsToContents()
        item =  QTableWidgetItem()
        self.result_out.setHorizontalHeaderItem(0, item)
        item =  QTableWidgetItem()
        self.result_out.setHorizontalHeaderItem(1, item)
        item =  QTableWidgetItem()
        self.result_out.setHorizontalHeaderItem(2, item)
        item =  QTableWidgetItem()
        self.result_out.setHorizontalHeaderItem(3, item)
        item =  QTableWidgetItem()
        self.result_out.setHorizontalHeaderItem(4, item)
        item =  QTableWidgetItem()
        self.result_out.setHorizontalHeaderItem(5, item)
        item =  QTableWidgetItem()
        self.result_out.setHorizontalHeaderItem(6, item)
        self.result_out.horizontalHeader().setDefaultSectionSize(100)
        self.result_out.horizontalHeader().setMinimumSectionSize(25)
        self.result_out.verticalHeader().setDefaultSectionSize(30)
        self.pushButton_2 =  QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(675, 390, 75, 21))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.p2_clicked)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar =  QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 509, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar =  QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)



        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    def p2_clicked(self):
        self.pyqt_clicked1.emit()
    def find_btn(self):
        self.pyqt_clicked2.emit()
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.check_Sid.setText(_translate("MainWindow", "学号", None))
        self.check_Sage.setText(_translate("MainWindow", "年龄自", None))
        self.check_Sname.setText(_translate("MainWindow", "姓名", None))
        self.check_Ssex.setText(_translate("MainWindow", "性别", None))
        self.label.setText(_translate("MainWindow", "到", None))
        self.check_Sdept.setText(_translate("MainWindow", "系", None))
        self.check_Sclass.setText(_translate("MainWindow", "班级", None))
        self.check_Saddr.setText(_translate("MainWindow", "地址", None))
        self.find.setText(_translate("MainWindow", "查询", None))
        self.sql_out.setText(self.sqlstring)
        item = self.result_out.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Sid", None))
        item = self.result_out.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Sname ", None))
        item = self.result_out.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Sage", None))
        item = self.result_out.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Ssex", None))
        item = self.result_out.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "Sclass", None))
        item = self.result_out.horizontalHeaderItem(5)
        item.setText(_translate("MainWindow", "Sdept", None))
        item = self.result_out.horizontalHeaderItem(6)
        item.setText(_translate("MainWindow", "Saddr", None))
        self.pushButton_2.setText(_translate("MainWindow", "退出", None))

    def mousePressEvent(self, event):
        # if event.KeyWord == Qt.LeftButton:
        print("nihao")

    def buttonTest(self):
        temp_sqlstring = self.sqlstring
        is_first = True
        if self.check_Sid.isChecked():
            mystr = self.Sid.text()
            if is_first:
                is_first = False
                if mystr.find("%") == -1:
                    temp_sqlstring += "Sid = '" + self.Sid.text() + "'"
                else:
                    temp_sqlstring += "Sid like '" + self.Sid.text() + "'"
            else:
                if mystr.find("%") == -1:
                    temp_sqlstring += " and Sid = '" + self.Sid.text() + "'"
                else:
                    temp_sqlstring += " and Sid like '" + self.Sid.text() + "'"

        if self.check_Sname.isChecked():
            if is_first:
                mystr = self.Sname.text()
                is_first = False
                if mystr.find("%") == -1:
                    temp_sqlstring += "Sname = '" + self.Sname.text() + "'"
                else:
                    temp_sqlstring += "Sname like '" + self.Sname.text() + "'"
            else:
                if mystr.find("%") == -1:
                    temp_sqlstring += " and Sname = '" + self.Sname.text() + "'"
                else:
                    temp_sqlstring += " and Sname like '" + self.Sname.text() + "'"

        if self.check_Sage.isChecked():
            if is_first:
                is_first = False
                temp_sqlstring += "Sage >= " + self.first_Sage.text() + \
                                  " and Sage <= " + self.last_Sage.text()
            else:
                temp_sqlstring += " and Sage >= " + self.first_Sage.text() + \
                                  " and Sage <= " + self.last_Sage.text()

        if self.check_Ssex.isChecked():
            if is_first:
                is_first = False
                temp_sqlstring += "Ssex = '" + self.Ssex.text() + "'"
            else:
                temp_sqlstring += " and Ssex = '" + self.Ssex.text() + "'"

        if self.check_Sclass.isChecked():
            if is_first:
                mystr = self.Sclass.text()
                is_first = False
                if mystr.find("%") == -1:
                    temp_sqlstring += "Sclass = '" + self.Sclass.text() + "'"
                else:
                    temp_sqlstring += "Sclass like '" + self.Sclass.text() + "'"
            else:
                if mystr.find("%") == -1:
                    temp_sqlstring += " and Sclass = '" + self.Sclass.text() + "'"
                else:
                    temp_sqlstring += " and Sclass like '" + self.Sclass.text() + "'"

        if self.check_Sdept.isChecked():
            if is_first:
                mystr = self.Sdept.text()
                is_first = False
                if mystr.find("%") == -1:
                    temp_sqlstring += "Sdept = '" + self.Sdept.text() + "'"
                else:
                    temp_sqlstring += "Sdept like '" + self.Sdept.text() + "'"
            else:
                if mystr.find("%") == -1:
                    temp_sqlstring += " and Sdept = '" + self.Sdept.text() + "'"
                else:
                    temp_sqlstring += " and Sdept like '" + self.Sdept.text() + "'"

        if self.check_Saddr.isChecked():
            if is_first:
                mystr = self.Saddr.text()
                is_first = False
                if mystr.find("%") == -1:
                    temp_sqlstring += "Saddr = '" + self.Saddr.text() + "'"
                else:
                    temp_sqlstring += " and Saddr like '" + self.Saddr.text() + "'"
            else:
                if mystr.find("%") == -1:
                    temp_sqlstring += " and Saddr = '" + self.Saddr.text() + "'"
                else:
                    temp_sqlstring += " and Saddr like '" + self.Saddr.text() + "'"

        self.result_out.clearContents()  # 每一次查询时清除表格中信息
        if not (is_first):
            print(temp_sqlstring)
            self.cur.execute(temp_sqlstring)
            k = 0
            for i in self.cur:
                print("----------",i)
                w = 0
                for j in i:
                    # 这里是将int类型转成string类型，方便后面文本设置
                    if type(j) == int:
                        newItem =  QTableWidgetItem(str(j))

                    else:
                        newItem =  QTableWidgetItem(j)
                    # 根据循环标签一次对table中的格子进行设置
                    self.result_out.setItem(k, w, newItem)
                    w += 1
                k += 1

        self.sql_out.setText("")
        self.sql_out.append(temp_sqlstring)
        print("find button pressed")

    def buttonExit(self):
        self.conn.commit()
        self.cur.close()
        self.conn.close()
        self.close()

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Escape:
            self.buttonExit()


class MyWindow( QMainWindow, Ui_MainWindow):
    pyqt_clicked1 = pyqtSignal()
    pyqt_clicked2 = pyqtSignal()
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)
        self.pyqt_clicked1.connect(self.buttonExit)
        self.pyqt_clicked2.connect(self.buttonTest)



if __name__ == "__main__":
    app =  QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    myshow = MyWindow()
    myshow.show()
    sys.exit(app.exec_())