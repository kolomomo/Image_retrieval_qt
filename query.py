# -*- coding: utf-8 -*-
# SECA
# MMAP 0.6610458234462783
# all MAP 0.8463228520542899
# P11 mean: [0.86051724 0.83014101 0.83921372 0.84627646 0.850826   0.85487125
#  0.85795868 0.86002889 0.86059498 0.85694303 0.5577796 ]
# CenterVLAD
# MMAP 0.64821052269954
# all MAP 0.8310386821166906
# P11 mean: [0.88149619 0.84223081 0.84204824 0.84317989 0.84760494 0.84817095
#  0.84099456 0.83963959 0.83207988 0.81459737 0.37172013]
# ResNet_CenterVLAD
# MMAP 0.5986843371324242
# all MAP 0.730441395048211
# P11 mean: [0.84626006 0.79932364 0.78371352 0.77119476 0.77373655 0.77184647
#  0.76563545 0.76064716 0.74556912 0.59116869 0.17291795]
# VGG16
# MMAP 0.30052564257767606
# all MAP 0.5287164117910352
# P11 mean: [0.82833139 0.6929997  0.64320407 0.60222626 0.56933041 0.53433035
#  0.49774529 0.45933488 0.41675877 0.35819627 0.17729793]
# ResNet50
# MMAP 0.4210623419102376
# all MAP 0.6840986163356793
# P11 mean: [0.83077925 0.7755499  0.75333949 0.73641523 0.72073668 0.70272319
#  0.68254442 0.65899625 0.62581328 0.57036853 0.23177303]

from retrieval.ir_query_v2 import *
from util.data import Database
#
from models.my_models import CenterVLAD
from keras.backend import l2_normalize, expand_dims
#0MMAP 0.6343613371830391 all MAP 0.8397149029347234
#1MMAP 0.5914800377582923 all MAP 0.8205283262664536
#MMAP 0.6564301422060463 all MAP 0.8487869585417427
# configs TODO
pick_layer = 'feature'    # 提取特征层名称
model_name = 'SEA_Net_X2'
model_path = '/home/wbo/PycharmProjects/Image_retrieval_qt/models/save_m/SEA_Net_X2.h5'
image_size = 128
COLOR_MODE = 'grayscale'  # 'grayscale'/'rgb'
d_type = 'cosine'     # 距离类型
depth = None        # 检索返回深度, None为返回全部, P11 需要设置depth=None
#
feature_path='./features'
Q_PAYH = '/home/wbo/Datasets/ir_test' # 待检索数据集
S_PATH = '/home/wbo/Datasets/ir_train' # '/home/wbo/Project/SEA/datasets/ir_train_rescale_128' # 检索数据集
save_Q_csv = '/home/wbo/PycharmProjects/Image_retrieval_qt/datasets/ir_test.csv'
save_S_csv = '/home/wbo/PycharmProjects/Image_retrieval_qt/datasets/ir_train.csv' #'/home/wbo/Project/SEA/datasets/ir_train_rescale_128.csv'
#

if model_name == 'CenterVLAD' or 'ResNet50_CenterVLAD':
    model = load_model(model_path, custom_objects={'CenterVLAD': CenterVLAD,
                                                   'l2_normalize':l2_normalize,
                                                   'expand_dims':expand_dims})
else:
    model = load_model(model_path)


qdb = Database(data_path=Q_PAYH, save_path=save_Q_csv)
sdb = Database(data_path=S_PATH, save_path=save_S_csv)
# 特征提取/加载
# 特征提取/加载
qsamlpe = feature_samples(
    model=model,
    model_name=model_name,
    image_size=image_size,
    pick_layer=pick_layer,
    db=qdb,
    dataset='QData',
    feature_path=feature_path,
    color_mode= COLOR_MODE,
                          )

ssamlpe = feature_samples(
    model=model,
    model_name=model_name,
    image_size=image_size,
    pick_layer=pick_layer,
    db=sdb,
    dataset='SData',
    feature_path=feature_path,
    color_mode= COLOR_MODE,
                          )

# 检索
classes = sdb.get_class()
ret = {c: [] for c in classes}
all_ap = []   # 所有查询ap值
P11 = []

QE = 0 # 拓展查询值

for k, query in enumerate(qsamlpe):
    print(query['cls'])
    if QE:
        print('QE：', QE)
        ap, ar, p11, result = query_infer(query, s_samples=ssamlpe, depth=depth, d_type=d_type)
        for i in range(QE):
            query['hist'] = result[i]['hist'] + query['hist']
        query['hist'] /= (QE + 1)
    ap, ar, p11, _ = query_infer(query, s_samples=ssamlpe, depth=depth, d_type=d_type)
    print('ap, ar, p11:', ap, ar, p11)
    all_ap.append(ap)
    P11.append(p11)
    ret[query['cls']].append(ap)
APs = ret
# 每一类的mAP
cls_MAPs = []
for cls, cls_APs in APs.items():
    if len(cls_APs):
        MAP = np.mean(cls_APs)
        print("Class {}, MAP {}".format(cls, MAP))
        cls_MAPs.append(MAP)
P11 = np.array(P11)

print("MMAP", np.mean(cls_MAPs))
print("all MAP", np.mean(all_ap))
print('P11 mean:', np.mean(P11, axis=0))
# np.save('/home/wbo/PycharmProjects/Image_retrieval_qt/features/P11/{}_P11'.format(model_name), np.mean(P11, axis=0))
print('finished')
