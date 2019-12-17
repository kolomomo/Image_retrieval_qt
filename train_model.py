import numpy as np
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
# from models.my_models import CenterVLAD_Net
# from models.pre_models import ResNet50_train
# from models.pre_models import VGG16_train
# from models.my_models import AttentionResNet56
# from models.my_models import ResNet50_CenterVLAD
# from models.my_models import SEA_Net_c
from models.my_models import SEA_Net_X
from util.data_gen import irdata_gen

# configs TODO
save_m = '/home/wbo/PycharmProjects/Image_retrieval_qt/models/save_m/SEA_Net_X1.h5' # 保存训练模型路径
save_h = '/home/wbo/PycharmProjects/Image_retrieval_qt/models/save_m/SEA_Net_X1_train_history' # 保存训练历史路径
train_data_csv = '/home/wbo/PycharmProjects/Image_retrieval_qt/datasets/ir_train.csv'
test_data_csv = '/home/wbo/PycharmProjects/Image_retrieval_qt/datasets/ir_test.csv'
IMAGE_SIZE = 128
BATCH = 32
CLASS_MODE = 'sparse' #'sparse'
COLOR_MODE = 'grayscale' #'grayscale'
# 模型
model = SEA_Net_X(DE=True, SE=False)
#

train_gen, val_gen = irdata_gen(
    train_csv=train_data_csv,
    img_size=IMAGE_SIZE,
    img_batch=BATCH,
    classmode=CLASS_MODE,
    colormode=COLOR_MODE,
)
print(model.summary())
## 学习率调整
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=10e-7, min_delta=0.01, verbose=1)
## 早停
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1)
## 保存最优模型
checkpoint = ModelCheckpoint(save_m, save_best_only=True)
#
callbacks= [early_stopper, checkpoint, lr_reducer]
# 训练
history = model.fit_generator(
                    train_gen,
                    validation_data=val_gen,
                    epochs=500,
                    class_weight='auto',
                    steps_per_epoch=train_gen.n // train_gen.batch_size+1,
                    validation_steps=val_gen.n // val_gen.batch_size+1,
                    callbacks=callbacks,
    )

# 保存训练历史记录
np.savez(save_h,
         loss = history.history['loss'],
         acc = history.history['acc'],
         val_loss = history.history['val_loss'],
         val_acc = history.history['val_acc'],
         )
