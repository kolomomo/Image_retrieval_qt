# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   FileName:    database
   Author:      kolomomo
   Date:        2020/8/12
   UpdateData:  2020/8/12:
-------------------------------------------------
   Description: 自定义数据集类
   
-------------------------------------------------
"""

import os
import pandas as pd


class Database(object):
    def __init__(self,
                 data_path='',
                 save_path=''):
        self.DB_csv = save_path
        self.DB_dir = data_path
        self.gen_csv()
        self.data = pd.read_csv(self.DB_csv)
        self.classes = set(self.data["cls"])

    def gen_csv(self):
        if os.path.exists(self.DB_csv):
            print('{}目录已存在!'.format(self.DB_csv))
            return None
        with open(self.DB_csv, 'w', encoding='UTF-8') as f:
            f.write("img,cls")
            for root, _, files in os.walk(self.DB_dir, topdown=False):
                # 平台环境
                if os.name == 'nt':
                    cls = root.split('\\')[-1]
                else:
                    cls = root.split('/')[-1]
                for name in files:
                    # if not name.endswith('.jpg'):
                    # continue
                    img = os.path.join(root, name)
                    f.write("\n{},{}".format(img, cls))
        print('csv已经生成!')

    def __len__(self):
        return len(self.data)

    def get_class(self):
        return self.classes

    def get_data(self):
        return self.data

    def get_class_nums(self,
                       class_name):
        return sum(self.data['cls'] == class_name)
