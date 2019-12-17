from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Dropout
#
from keras.applications.resnet50 import ResNet50
#
from keras.models import Model, load_model
from keras.regularizers import l2
from models.CenterVLAD_layer import CenterVLAD
from keras.backend import l2_normalize, expand_dims
from keras.optimizers import Adam
from models.blocks import *

import tensorflow as tf
import os

#
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.backend.tensorflow_backend import set_session,clear_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3 # 分配显存使用比例
set_session(tf.Session(config=config))
clear_session()
#

def AttentionResNet56(shape=(128, 128, 3), n_channels=64, n_classes=116,
                      dropout=0.5, regularization=0.01):
    """
    Attention-56 ResNet
    https://arxiv.org/abs/1704.06904
    """
    regularizer = l2(regularization)

    input_ = Input(shape=shape)
    x = Conv2D(n_channels, (7, 7), strides=(2, 2), padding='same')(input_) # 112x112 64
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

    pool_size = (x.get_shape()[1].value, x.get_shape()[2].value)
    x = AveragePooling2D(pool_size=pool_size, strides=(1, 1))(x)
    x = Flatten(name='feature')(x)
    if dropout:
        x = Dropout(dropout)(x)
    output = Dense(n_classes, kernel_regularizer=regularizer, activation='softmax')(x)

    model = Model(input_, output)

    adam = Adam(lr=0.0001)
    # 编译
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

def centerVLAD2(shape=(128, 128, 1), n_channels=32, n_classes=116):
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

def SEA_Net_a(shape=(128, 128, 1), n_channels=32, n_classes=116):
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

    x = attention_block_a(x, encoder_depth=3)
    x = attention_block_a(x, encoder_depth=3)
    x = attention_block_a(x, encoder_depth=3)

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)  # 8

    x = GlobalAveragePooling2D(name='feature')(x)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input_, output)

    adam = Adam(lr=0.0001)
    # 编译
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

def SEA_Net_b(shape=(128, 128, 1), n_channels=32, n_classes=116):
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

    x = attention_block_b(x, encoder_depth=3)
    x = attention_block_b(x, encoder_depth=3)
    x = attention_block_b(x, encoder_depth=3)

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)  # 8

    x = GlobalAveragePooling2D(name='feature')(x)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input_, output)

    adam = Adam(lr=0.0001)
    # 编译
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

def SEA_Net_c(shape=(128, 128, 1), n_channels=32, n_classes=116):
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

    x = GlobalAveragePooling2D(name='feature')(x)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input_, output)

    adam = Adam(lr=0.0001)
    # 编译
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

def SEA_Net_X(shape=(128, 128, 1), n_channels=32, n_classes=116, DE=False, SE=False, dropout=0):
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

    x = attention_block_X(x, encoder_depth=3, include_de=DE, include_se=SE)
    x = attention_block_X(x, encoder_depth=3, include_de=DE, include_se=SE)
    x = attention_block_X(x, encoder_depth=3, include_de=DE, include_se=SE)

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)  # 8

    x = GlobalAveragePooling2D()(x)
    x = Lambda(lambda a: l2_normalize(a, axis=-1), name='feature')(x)
    if dropout:
        x = Dropout(dropout)(x)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input_, output)

    adam = Adam(lr=0.0001)
    # 编译
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

def SECA_Net(shape=(128, 128, 1), n_channels=32, n_classes=116, lambda_center=0.02):
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

    x = GlobalAveragePooling2D(name='feature')(x)

    print(x.shape)

    output = Dense(n_classes, activation='softmax', name='predict')(x)

    feature = x
    featrue_size = int(feature.shape[1])

    input_target = Input(shape=(1,))
    center = Embedding(n_classes, featrue_size)(input_target)
    loss_l4 = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='loss_l4')([feature, center])

    model_train = Model(inputs=[input_, input_target], outputs=[output, loss_l4])
    model_predict = Model(input_, output)

    adam = Adam(lr=0.0001)
    lambda_c = lambda_center
    model_train.compile(optimizer=adam, loss=['sparse_categorical_crossentropy', lambda y_true, y_pred: y_pred],
                  loss_weights=[1, lambda_c], metrics=['accuracy'])

    return model_train, model_predict

def CenterVLAD_Net(shape=(128, 128, 1), n_channels=32, n_classes=116, dropout=0.5):
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
    if dropout:
        x = Dropout(dropout)(x)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_, outputs=output)

    adam = Adam(lr=0.0001)
    # 编译

    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

def ResNet50_CenterVLAD(n_classes=116, dropout=0.5):
    base_model = ResNet50(weights=None, input_shape=(128,128,3), include_top=False)
    x = base_model.output
    x = Conv2D(32, (1,1), padding='same')(x)
    x = Lambda(lambda a: l2_normalize(a, axis=-1), name='lambda1')(x)
    x = CenterVLAD(num_clusters=n_classes)(x)
    x = Lambda(lambda a: l2_normalize(a, axis=-1), name='feature')(x)
    if dropout:
        x = Dropout(dropout)(x)
    output = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    adam = Adam(lr=0.0001)
    # 编译
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

def CNNs(shape=(128, 128, 3), n_classes=116):
    input_ = Input(shape=shape)
    x = Conv2D(64, (11, 11), padding='same')(input_)
    x = MaxPool2D(pool_size=(3, 3), strides=2)(x)  # 32
    x = Dropout(0.6)(x)
    x = Conv2D(98, (5, 5), padding='same')(x)
    x = MaxPool2D(pool_size=(3, 3),strides=2)(x)  # 16
    x = Conv2D(192, (5, 5), padding='same')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = MaxPool2D(pool_size=(3, 3),strides=2)(x)  # 16
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Flatten()(x)
    x = Dense(2048, name='feature')(x)
    x = Dense(2048)(x)
    x = Dense(2048)(x)
    output = Dense(n_classes, activation='softmax')(x)
    model = Model(input_, output)

    adam = Adam(lr=0.0001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

if __name__=='__main__':
    model= CenterVLAD_Net()
    print(model.summary())


