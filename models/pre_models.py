from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.optimizers import Adam
# 显卡选择
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 分配显存使用比例
set_session(tf.Session(config=config))
#

def ResNet50_train(classes=116):
    base_model = ResNet50(weights=None, input_shape=(128,128,3), include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D(name='feature')(x)
    x = Dense(classes, name='predictions', activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    adam = Adam(lr=0.0001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

def VGG16_train(classes=116):
    base_model = VGG16(weights=None, input_shape=(128,128,3), include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D(name='feature')(x)
    x = Dense(classes, name='predictions', activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    adam = Adam(lr=0.0001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model