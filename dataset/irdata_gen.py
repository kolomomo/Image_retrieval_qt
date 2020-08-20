# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   FileName:    irdata_gen
   Author:      kolomomo
   Date:        2020/8/13
   UpdateData:  2020/8/13:
-------------------------------------------------
   Description: 数据集生成器
   
-------------------------------------------------
"""

import pandas as pd
from keras.preprocessing.image import ImageDataGenerator


def irdata_gen(train_csv,
               test_csv=None,
               img_size=128,
               img_batch=32,
               split=0.2,
               classmode='sparse',
               colormode='grayscale'):

    """

    Args:
        train_csv:  训练数据集csv文件
        test_csv:   测试数据集csv文件
        img_size:   图像大小
        img_batch:  生成器batch size
        split:  划分验证集比例
        classmode:  类别标签类型
        colormode:  图像颜色类型

    Returns:

    """

    # 可选数据增强，参考keras文档
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=split,  # 验证集划分
    )

    image_size = img_size
    batch_size = img_batch

    data_s = pd.read_csv(train_csv)
    data_s.columns = ['x_col', 'y_col']
    data_s = data_s.sample(frac=1)  # 打乱,重要

    train_gen = datagen.flow_from_dataframe(dataframe=data_s,
                                            directory=None,
                                            target_size=(image_size, image_size),
                                            color_mode=colormode,
                                            x_col='x_col',
                                            y_col='y_col',
                                            batch_size=batch_size,
                                            shuffle=True,
                                            class_mode=classmode,  # 'sparse',
                                            subset='training')

    valid_gen = datagen.flow_from_dataframe(dataframe=data_s,
                                            directory=None,
                                            target_size=(image_size, image_size),
                                            color_mode=colormode,
                                            x_col='x_col',
                                            y_col='y_col',
                                            batch_size=batch_size,
                                            shuffle=True,
                                            class_mode=classmode,  # 'sparse',
                                            subset='validation')

    if test_csv is not None:
        datagen_test = ImageDataGenerator(
            rescale=1. / 255,
        )
        data_test = pd.read_csv(test_csv)
        data_test.columns = ['x_col', 'y_col']
        test_gen = datagen_test.flow_from_dataframe(dataframe=data_test,
                                                    directory=None,
                                                    target_size=(image_size, image_size),
                                                    color_mode=colormode,
                                                    x_col='x_col',
                                                    y_col='y_col',
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    class_mode=classmode,)

        return train_gen, valid_gen, test_gen
    else:
        return train_gen, valid_gen
