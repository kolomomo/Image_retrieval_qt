# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   FileName:    train_model
   Author:      kolomomo
   Date:        2020/8/13
   UpdateData:  2020/8/13:
-------------------------------------------------
   Description: 训练模型
   
-------------------------------------------------
"""

import argparse

import os
import yaml
import numpy as np
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from models.load_models import load_irmodels
from dataset.database import Database
from dataset.irdata_gen import irdata_gen
from keras.optimizers import Adam


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/default.yaml', help='*.cfg path')
    opt = parser.parse_args()

    with open(opt.cfg, encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)  # model dict

    os.chdir(cfg['root_dir'])
    experiment_dir = os.path.join(cfg['root_dir'], cfg['experiment_name'])

    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    train_data_csv = os.path.join(experiment_dir, 'ir_train.csv')
    test_data_csv = os.path.join(experiment_dir, 'ir_test.csv')

    Database(data_path=cfg['ir_train_dir'], save_path=train_data_csv)
    Database(data_path=cfg['ir_test_dir'], save_path=test_data_csv)

    save_m = os.path.join(experiment_dir, cfg['model_name']+'.h5')  # 保存训练模型路径
    save_h = os.path.join(experiment_dir, cfg['model_name']+'_train_history')  # 保存训练历史路径

    IMAGE_SIZE = cfg['image_size']    # 修改model中的尺寸
    BATCH = cfg['batch_size']
    EPOCH = cfg['epoch']
    CLASS_MODE = cfg['class_mode']
    COLOR_MODE = cfg['color_mode']
    LR = cfg['lr']
    # 模型
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['gputouse'])

    adam = Adam(lr=LR)

    model = load_irmodels(cfg)
    model.compile(optimizer=adam,
                  loss=cfg['loss'],
                  metrics=['accuracy'])

    train_gen, val_gen = irdata_gen(train_csv=train_data_csv,
                                    img_size=IMAGE_SIZE,
                                    img_batch=BATCH,
                                    classmode=CLASS_MODE,
                                    colormode=COLOR_MODE,)

    print(model.summary())

    # 学习率调整
    lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.2,
                                   patience=7,
                                   min_lr=10e-7,
                                   min_delta=0.01,
                                   verbose=1)
    # 早停
    early_stopper = EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=20,
                                  verbose=1)
    # 保存最优模型
    checkpoint = ModelCheckpoint(save_m,
                                 save_best_only=True)
    # 回调
    callbacks = [early_stopper, checkpoint, lr_reducer]
    # 训练
    history = model.fit_generator(train_gen,
                                  validation_data=val_gen,
                                  epochs=EPOCH,
                                  class_weight='auto',
                                  steps_per_epoch=train_gen.n // train_gen.batch_size,
                                  validation_steps=val_gen.n // val_gen.batch_size,
                                  callbacks=callbacks,)

    # 保存训练历史记录
    print(history.history)
    np.savez(save_h,
             loss=history.history['loss'],
             acc=history.history['accuracy'],
             val_loss=history.history['val_loss'],
             val_acc=history.history['val_accuracy'],)
