# 数据集生成器——多通道
import keras
from util.preprocessing.image import ImageDataGenerator
import pandas as pd

def irdata_gen(train_csv,
               test_csv=None,
               img_size=128,
               img_batch=32,
               split=0.2,
               classmode='sparse',
               colormode='grayscale'):
    '''
    :param img_csv: csv
    :param img_size: size
    :param img_batch: batch
    :return: generator of training/validation/test
    '''

    datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=split, # 验证集划分
    )
    IMAGE_SIZE = img_size
    BATCH_SIZE = img_batch

    data_s = pd.read_csv(train_csv)
    data_s.columns = ['x_col', 'y_col']
    data_s = data_s.sample(frac=1)  # 打乱,重要

    train_gen = datagen.flow_from_dataframe(dataframe=data_s,
                                            directory=None,
                                            target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                            color_mode=colormode,
                                            x_col='x_col',
                                            y_col='y_col',
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            class_mode=classmode,  # 'sparse',
                                            subset='training')

    valid_gen = datagen.flow_from_dataframe(dataframe=data_s,
                                            directory=None,
                                            target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                            color_mode=colormode,
                                            x_col='x_col',
                                            y_col='y_col',
                                            batch_size=BATCH_SIZE,
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
                                                target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                color_mode=colormode,
                                                x_col='x_col',
                                                y_col='y_col',
                                                batch_size=BATCH_SIZE,
                                                shuffle=False,
                                                class_mode= classmode, #'sparse',
                                                )

        return train_gen, valid_gen, test_gen
    else:
        return train_gen, valid_gen
