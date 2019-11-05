#SECA
#    accuracy                           0.84      1000
#    macro avg       0.66      0.63      0.63      1000
# weighted avg       0.84      0.84      0.83      1000
#CenterVLAD
#    accuracy                           0.85      1000
#    macro avg       0.65      0.66      0.64      1000
# weighted avg       0.84      0.85      0.84      1000
# ResNet50+CenterVLAD
#    accuracy                           0.84      1000
#    macro avg       0.63      0.64      0.61      1000
# weighted avg       0.84      0.84      0.83      1000
#VGG16
#   accuracy                           0.77      1000
#    macro avg       0.56      0.54      0.52      1000
# weighted avg       0.78      0.77      0.76      1000
#ResNet50
#    accuracy                           0.80      1000
#    macro avg       0.58      0.58      0.56      1000
# weighted avg       0.80      0.80      0.79      1000

from keras.models import load_model, Model
from keras.preprocessing import image
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import tensorflow as tf
#
from models.my_models import CenterVLAD
from keras.backend import l2_normalize, expand_dims
#
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 分配显存使用比例
set_session(tf.Session(config=config))
#

# 分类评估
IMAGE_SIZE = 128
BATCH = 32
CLASS_MODE = 'categorical' #'sparse'
COLOR_MODE = 'rgb' #'grayscale'
val_path = '/home/wbo/Datasets/ir_test' # 测试集
model_n = '/home/wbo/PycharmProjects/Image_retrieval_qt/models/save_m/ResNet50.h5' # 模型路径
#

datagen_test = image.ImageDataGenerator(
   rescale=1./255,
)


test_gender = datagen_test.flow_from_directory(
    val_path,
    batch_size=BATCH,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    shuffle=False,
    class_mode=CLASS_MODE,
    color_mode=COLOR_MODE,
)

# CenterVLAD模型
# model=load_model(model_n,
#                  custom_objects={'CenterVLAD': CenterVLAD, 'l2_normalize':l2_normalize, 'expand_dims':expand_dims})

model = load_model(model_n)
print(model.summary())
result = model.predict_generator(test_gender, steps = test_gender.n // test_gender.batch_size+1)

result1 = test_gender.classes
result2 = np.argmax(result, axis=1)

# APs, ARs = eva_ap_ar(result2, result1)
# AP = np.mean(APs)
# AR = np.mean(ARs)
# F1 = 2*AP*AR/(AP+AR)
# print('AP:{0:.4f},  AR:{1:.4f}, F1 score:{2:.4f}'.format(AP, AR, F1))

print(classification_report(result1,result2))
print(sum((result2-result1) == 0))
