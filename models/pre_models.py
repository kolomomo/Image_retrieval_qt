from keras.layers import Flatten
#
from keras.models import Model, load_model
from keras.regularizers import l2
from models.CenterVLAD_layer import CenterVLAD
from keras.backend import l2_normalize, expand_dims
# import sys
# sys.path.append('.')
from models.blocks import *

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 分配显存使用比例
set_session(tf.Session(config=config))
#
def centerVLAD(shape=(128, 128, 1), n_channels=32, n_classes=116):
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
    #
    # x = Conv2D(32, (3, 3), padding='same')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = MaxPool2D(pool_size=(2, 2))(x)  # 4

    x = attention_block_c(x, encoder_depth=2)
    # x = attention_block_c(x, encoder_depth=2)
    # x = attention_block_c(x, encoder_depth=2)

    x = Lambda(lambda a: l2_normalize(a,axis=-1), name='lambda1')(x)

    x = CenterVLAD(num_clusters=n_classes)(x)

    x = Lambda(lambda a: expand_dims(a,axis=1), name='lambda2')(x)
    x = Lambda(lambda a: expand_dims(a,axis=1), name='lambda3')(x)
    # x = conv2d_bn(x, 4096, 1, 1)

    x = Flatten()(x)

    # x = Conv2D(4096, kernel_size=(1, 1), padding='same')(x)
    # x = GlobalAveragePooling2D()(x)

    x = Lambda(lambda a: l2_normalize(a,axis=-1), name='feature')(x)

    output = Dense(n_classes, activation='softmax')(x)
    model = Model(input_, output)
    return model


def SEA(shape=(128, 128, 1), n_channels=32, n_classes=116):
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

    x = attention_block_c(x, encoder_depth=3)
    x = attention_block_c(x, encoder_depth=3)
    x = attention_block_c(x, encoder_depth=3)

    x = GlobalAveragePooling2D(name='feature')(x)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input_, output)
    return model

def CenterVLAD_PLUS(shape=(128, 128, 1), n_channels=32, n_classes=116):
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

    x = attention_block_c(x, encoder_depth=3)
    x = attention_block_c(x, encoder_depth=3)
    x = attention_block_c(x, encoder_depth=3)

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)  # 8


    x = Lambda(lambda a: l2_normalize(a, axis=-1), name='lambda1')(x)  # 22

    x = CenterVLAD(num_clusters=n_classes)(x)  # 23
    # # PCA
    # x = Lambda(lambda a: expand_dims(a, axis=1), name='lambda2')(x)  # 24
    # x = Lambda(lambda a: expand_dims(a, axis=1), name='lambda3')(x)  # 25
    #
    # x = Conv2D(2048, (1, 1), padding='same')(x)
    # x = Flatten()(x)  # 27

    x = Lambda(lambda a: l2_normalize(a, axis=-1), name='feature')(x)  # 27

    output = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_, outputs=output)
    return model

if __name__== '__main__':
    model= CenterVLAD_PLUS()
    print(model.summary())
