
from models.my_models import SECA_Net
from keras.models import load_model
from util.data_gen import irdata_gen

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import numpy as np

# 01
save_m = '/home/wbo/PycharmProjects/Image_retrieval_qt/models/save_m/SECA.h5' # 保存训练模型路径
save_h =  '/home/wbo/PycharmProjects/Image_retrieval_qt/models/save_m/SECA_train_history' # 保存训练历史路径
train_data_csv = '/home/wbo/PycharmProjects/Image_retrieval_qt/datasets/ir_train.csv'
test_data_csv = '/home/wbo/PycharmProjects/Image_retrieval_qt/datasets/ir_test.csv'
IMAGE_SIZE = 128
BATCH = 32
CLASS_MODE = 'sparse'
COLOR_MODE = 'grayscale'

train_gen, val_gen = irdata_gen(
    train_csv=train_data_csv,
    img_size=IMAGE_SIZE,
    img_batch=BATCH,
    classmode=CLASS_MODE,
    colormode=COLOR_MODE,
)

# 数据集输入输出增加 center
def gender_new(gen):
    while gen.next():
        A = gen.next()
        ramdomy = np.ones((len(A[1]),1))
        yield ([A[0],A[1]], [A[1], ramdomy])

tg = gender_new(train_gen)
vg = gender_new(val_gen)

# 模型
model,model_p = SECA_Net(shape=(128,128,1), n_classes=116, lambda_center=0.002)
print(model.summary())
# prepare usefull callbacks
## 学习率调整
lr_reducer = ReduceLROnPlateau(monitor='val_predict_loss', factor=0.2, patience=7, min_lr=10e-7, min_delta=0.01, verbose=1)
## 早停
early_stopper = EarlyStopping(monitor='val_predict_loss', min_delta=0, patience=20, verbose=1)
## 保存最优模型
checkpoint = ModelCheckpoint(save_m, save_best_only=True)

callbacks= [early_stopper, lr_reducer]

# 训练
history = model.fit_generator(
                    tg,
                    validation_data=vg,
                    epochs=500,
                    # class_weight='auto',
                    steps_per_epoch=train_gen.n // train_gen.batch_size+1,
                    validation_steps=val_gen.n // val_gen.batch_size+1,
                    callbacks=callbacks,
    )

model_p.save(save_m)

# 保存训练历史记录
np.savez(save_h,
         loss = history.history['predict_loss'],
         acc = history.history['predict_acc'],
         all_loss = history.history['loss'],
         val_loss = history.history['val_predict_loss'],
         val_acc = history.history['val_predict_acc'],
         )
