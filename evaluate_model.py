# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   FileName:    evaluate_model
   Author:      kolomomo
   Date:        2020/8/13
   UpdateData:  2020/8/13:
-------------------------------------------------
   Description:
   
-------------------------------------------------
"""

import argparse

import os
import yaml
import numpy as np

from keras.preprocessing import image
from sklearn.metrics import classification_report
from models.load_models import load_irmodels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/default.yaml', help='*.cfg path')
    opt = parser.parse_args()

    with open(opt.cfg, encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['gputouse'])

    IMAGE_SIZE = cfg['image_size']  # 修改model中的尺寸
    BATCH = cfg['batch_size']

    CLASS_MODE = cfg['class_mode']
    COLOR_MODE = cfg['color_mode']

    val_path = cfg['ir_val_dir']  # 测试集

    experiment_dir = os.path.join(cfg['root_dir'], cfg['experiment_name'])
    model_path = os.path.join(experiment_dir, cfg['model_name'] + '.h5')
    resulte_csv = os.path.join(experiment_dir, cfg['model_name'] + '_evaluate.txt')  # 保存评估结果文件

    datagen_test = image.ImageDataGenerator(
        rescale=1. / 255,
    )

    test_gender = datagen_test.flow_from_directory(
        val_path,
        batch_size=BATCH,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        shuffle=False,
        class_mode=CLASS_MODE,
        color_mode=COLOR_MODE,
    )

    model = load_irmodels(cfg)

    print(model.summary())
    result = model.predict_generator(test_gender, steps=test_gender.n // test_gender.batch_size + 1)

    result1 = test_gender.classes
    result2 = np.argmax(result, axis=1)

    # APs, ARs = eva_ap_ar(result2, result1)
    # AP = np.mean(APs)
    # AR = np.mean(ARs)
    # F1 = 2*AP*AR/(AP+AR)
    # print('AP:{0:.4f},  AR:{1:.4f}, F1 score:{2:.4f}'.format(AP, AR, F1))
    rows = classification_report(result1, result2)
    print(rows)
    with open(resulte_csv, "w", newline='') as f:
        f.writelines(rows)
