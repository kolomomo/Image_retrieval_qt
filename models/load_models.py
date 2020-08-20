# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   FileName:    load_models
   Author:      kolomomo
   Date:        2020/8/13
   UpdateData:  2020/8/13:
-------------------------------------------------
   Description: 选择加载不同模型
   
-------------------------------------------------
"""

from models.nets import *


def load_irmodels(cfg):
    if cfg['model_name'] == 'CenterVLAD_Net':
        model_o = CenterVLAD_Net(shape=cfg['input_shape'],
                                 n_channels=cfg['n_channels'],
                                 n_classes=cfg['n_classes'],
                                 dropout=cfg['dropout'], )

        return model_o
    elif cfg['model_name'] == 'AttentionResNet56':
        model_o = AttentionResNet56(shape=cfg['input_shape'],
                                    n_channels=cfg['n_channels'],
                                    n_classes=cfg['n_classes'],
                                    dropout=cfg['dropout'],)
        return model_o
    else:
        print('未定义模型：', cfg['model_name'])
        return None
