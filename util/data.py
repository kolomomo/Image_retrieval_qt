# -*- coding: utf-8 -*-

import os
import pandas as pd
#             #
# 自定义数据集类 #
#             #
class Database(object):
  '''
  将数据集保存为csv
  '''
  def __init__(self, data_path, save_path):
    self.DB_csv = save_path
    self.DB_dir = data_path
    self._gen_csv()
    self.data = pd.read_csv(self.DB_csv)
    self.classes = set(self.data["cls"])

  def _gen_csv(self):
    if os.path.exists(self.DB_csv):
      return
    with open(self.DB_csv, 'w', encoding='UTF-8') as f:
      f.write("img,cls")
      for root, _, files in os.walk(self.DB_dir, topdown=False):
        cls = root.split('/')[-1]
        for name in files:
          # if not name.endswith('.jpg'):
          #   continue
          img = os.path.join(root, name)
          f.write("\n{},{}".format(img, cls))
    print('Finished!')

  def __len__(self):
    return len(self.data)

  def get_class(self):
    return self.classes

  def get_data(self):
    return self.data

  def get_class_nums(self, classname):
    return sum(self.data['cls'] == classname)