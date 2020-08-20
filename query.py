# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   FileName:    query
   Author:      kolomomo
   Date:        2020/8/13
   UpdateData:  2020/8/13:
-------------------------------------------------
   Description: 使用神经网络提取离线特征并进行图像检索
   
-------------------------------------------------
"""

import argparse

import yaml
from retrieval.ir_query import *
from dataset.database import Database
from models.load_models import load_irmodels
from keras.models import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/default.yaml', help='*.cfg path')
    opt = parser.parse_args()

    with open(opt.cfg, encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)  # model dict

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['gputouse'])

    pick_layer = cfg['pick_layer']    # 提取特征层名称
    model_name = cfg['model_name']
    experiment_dir = os.path.join(cfg['root_dir'], cfg['experiment_name'])
    model_path = os.path.join(experiment_dir, cfg['model_name']+'.h5')

    image_size = cfg['image_size']
    COLOR_MODE = cfg['color_mode']
    d_type = cfg['d_type']    # 距离类型
    depth = cfg['depth']      # 检索返回深度, None为返回全部, P11 需要设置depth=None

    feature_path = os.path.join(experiment_dir, 'features')
    if not os.path.exists(feature_path):
        os.mkdir(feature_path)

    Q_PATH = cfg['q_path']  # 待检索数据集
    S_PATH = cfg['s_path']  # 检索数据集
    q_name = cfg['q_name']
    s_name = cfg['s_name']
    save_Q_csv = os.path.join(experiment_dir, 'ir_Q.csv')
    save_S_csv = os.path.join(experiment_dir, 'ir_S.csv')
    save_result_txt = os.path.join(experiment_dir, 'ir_query_result.txt')

    qdb = Database(data_path=Q_PATH, save_path=save_Q_csv)
    sdb = Database(data_path=S_PATH, save_path=save_S_csv)

    model = load_irmodels(cfg)
    feat_model = Model(inputs=model.input, outputs=model.get_layer(pick_layer).output)

    print(feat_model.summary())

    # 特征提取/加载
    qsamlpe = feature_samples(model_name=model_name,
                              pick_layer=pick_layer,
                              model=feat_model,
                              db=qdb,
                              image_size=image_size,
                              color_mode=COLOR_MODE,
                              dataset=q_name,
                              feature_path=feature_path,)

    ssamlpe = feature_samples(model_name=model_name,
                              pick_layer=pick_layer,
                              model=feat_model,
                              db=sdb,
                              image_size=image_size,
                              color_mode=COLOR_MODE,
                              dataset=s_name,
                              feature_path=feature_path,)

    # 检索
    classes = sdb.get_class()
    ret = {c: [] for c in classes}
    all_ap = []   # 所有查询ap值
    P11 = []

    QE = cfg['qe']  # 拓展查询值

    for k, query in enumerate(qsamlpe):
        if QE:
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

    with open(save_result_txt, "w", newline='') as f:
        for cls, cls_APs in APs.items():
            if len(cls_APs):
                MAP = np.mean(cls_APs)
                print("Class {}, MAP {}".format(cls, MAP))
                f.write("Class {}, MAP {} \n".format(cls, MAP))
                cls_MAPs.append(MAP)
        P11 = np.array(P11)

        print("MMAP", np.mean(cls_MAPs))
        f.write("MMAP {} \n".format(np.mean(cls_MAPs)))
        print("all MAP", np.mean(all_ap))
        f.write("all MAP {} \n".format(np.mean(all_ap)))
        print('P11 mean:', np.mean(P11, axis=0))
        f.write("P11 mean {} \n".format(np.mean(P11, axis=0)))

        np.save(experiment_dir + '/{}_P11'.format(model_name), np.mean(P11, axis=0))
        print('finished')
