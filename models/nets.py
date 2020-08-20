# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   FileName:    nets
   Author:      kolomomo
   Date:        2020/8/12
   UpdateData:  2020/8/12:
-------------------------------------------------
   Description: 各种模型结构定义
   
-------------------------------------------------
"""

from keras.models import Model
from keras.regularizers import l2
from keras.backend import l2_normalize

from models.blocks import *
from models.centervlad_layer import CenterVLAD


def AttentionResNet56(shape=(128, 128, 3),
                      n_channels=64,
                      n_classes=116,
                      dropout=0.5,
                      regularization=0.01):
    """
    Attention-56 ResNet
    https://arxiv.org/abs/1704.06904
    """
    regularizer = l2(regularization)

    input_ = Input(shape=shape)
    x = Conv2D(n_channels, (7, 7), strides=(2, 2), padding='same')(input_)  # 112x112 64
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)  # 56x56 32

    x = residual_block(x, output_channels=n_channels * 4)  # 56x56 32
    x = attention_block(x, encoder_depth=3)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 8, stride=2)  # 28x28 16
    x = attention_block(x, encoder_depth=2)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 16, stride=2)  # 14x14 8
    x = attention_block(x, encoder_depth=1)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 32, stride=2)  # 7x7 4
    x = residual_block(x, output_channels=n_channels * 32)
    x = residual_block(x, output_channels=n_channels * 32)

    pool_size = (x.get_shape()[1], x.get_shape()[2])
    x = AveragePooling2D(pool_size=pool_size, strides=(1, 1))(x)
    x = Flatten(name='feature')(x)
    if dropout:
        x = Dropout(dropout)(x)
    output = Dense(n_classes, kernel_regularizer=regularizer, activation='softmax')(x)

    at56_model = Model(input_, output)

    return at56_model


def CenterVLAD_Net(shape=(128, 128, 1),
                   n_channels=32,
                   n_classes=116,
                   dropout=0.5):
    input_ = Input(shape=shape)
    x = Conv2D(n_channels, (5, 5), padding='same')(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)  # 64

    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)  # 32

    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)  # 16

    x = attention_block_se(x, encoder_depth=3)
    x = attention_block_se(x, encoder_depth=3)
    x = attention_block_se(x, encoder_depth=3)

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Lambda(lambda a: l2_normalize(a, axis=-1), name='lambda1')(x)

    x = CenterVLAD(num_clusters=n_classes)(x)

    x = Lambda(lambda a: l2_normalize(a, axis=-1), name='feature')(x)
    if dropout:
        x = Dropout(dropout)(x)
    output = Dense(n_classes, activation='softmax')(x)

    centervald_model = Model(inputs=input_, outputs=output)

    return centervald_model


if __name__ == '__main__':
    model = CenterVLAD_Net()
    for l in model.layers:
        print(l.name)
