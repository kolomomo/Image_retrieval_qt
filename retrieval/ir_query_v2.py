import numpy as np
from scipy import spatial
from six.moves import cPickle
from keras.models import Model, load_model
from keras.preprocessing import image
import os

def distance(v1, v2, d_type='d1'):
  '''各种距离
    欧斯距离： d2
    标准欧距: d2-norm
    曼哈顿距离： d1
    余弦距离： cosine
  '''

  assert v1.shape == v2.shape, "shape of two vectors need to be same!"

  if d_type == 'd1':
    return np.sum(np.absolute(v1 - v2))
  elif d_type == 'd2':
    return np.sum((v1 - v2) ** 2)
  elif d_type == 'd2-norm':
    return 2 - 2 * np.dot(v1, v2)
  elif d_type == 'd3':
    pass
  elif d_type == 'hausdorff':
    v1=np.expand_dims(v1, 0)
    v2=np.expand_dims(v2, 0)
    return spatial.distance.directed_hausdorff(v1, v2)[0]
  elif d_type == 'd5':
    pass
  elif d_type == 'd6':
    pass
  elif d_type == 'd7':
    pass
  elif d_type == 'd8':
    pass
  elif d_type == 'cosine':
    return spatial.distance.cosine(v1, v2)
  elif d_type == 'square':
    return np.sum((v1 - v2) ** 2)

# 计算检索AP/AP/P11
def AP(label, results, sort=True, P11=True):
  ''' infer a query, return it's ap

    arguments
      label  : 查询label
      results: 字典，包含3个键
               {
                 'dis': <距离 ample & query>,
                 'cls': <类别/标签>
                 'cns': <cls 类别图像个数>
               }
      sort   : sort the results by distance
      P11   : return P11
  '''
  if sort:
    results = sorted(results, key=lambda x: x['dis'])
  precision = []
  recall = []
  hit = 0

  for i, result in enumerate(results):
    if result['cls'] == label:
      hit += 1
      precision.append(hit / (i+1.))
      recall.append(hit / result['cns'])

  if hit == 0:
    if P11:
      return 0.,0.,[0. for i in range(11)]
    else:
      return 0.,0.

  if P11:
    P = []
    for k in range(0, 11):
      flag = 1
      for idxv, v in enumerate(recall):
        if v >= k / 10 and flag:
          P.append(precision[idxv])
          flag = 0

    return precision, recall, P
  else:
    return precision, recall

# 单个检索
def query_infer(q_sample, s_samples, depth=10, d_type='d1', P11=True):
  '''
  :param q_sample: 查询图像
  :param s_samples:检索库
  :param depth: 检索深度
  :param d_type: 检索距离测度
  :return: ap,ar,P11 和 检索排序结果result
  '''
  q_img, q_cls, q_hist = q_sample['img'], q_sample['cls'], q_sample['hist']
  results = []

  for idx, sample in enumerate(s_samples):
    s_img, s_cls, s_hist, s_cns = sample['img'], sample['cls'], sample['hist'], sample['cns']
    if q_img == s_img: # 相同的排除
      continue
    results.append({
                    'dis': distance(q_hist, s_hist, d_type=d_type),
                    'cls': s_cls,
                    'img': s_img,
                    'cns': s_cns,
                    'hist': s_hist
                  })

  results = sorted(results, key=lambda x: x['dis']) # 结果按距离排序

  if depth and depth <= len(results): # 返回深度
    results = results[:depth]
  p, r, p11 = AP(q_cls, results, sort=False, P11=P11)
  ap = np.mean(p)
  ar = np.mean(r)
  return ap, ar, p11, results

# 特征提取
def feature_samples(model, model_name, pick_layer, db,
                    image_size=128, color_mode='grayscale',
                    dataset='QData', feature_path='./features', verbose=True):
  '''
  :param model: 提取特征model
  :param model_name: model名称
  :param pick_layer: 提取特征的layer
  :param db: Dataset类, 特征库图像
  :param image_size: 图像大小
  :param color_mode: 图像类型（grayscale/rgb...）
  :param dataset:  区分检索|待检索特征（Q|S）
  :param feature_path: 特征保存路径
  :param verbose: 输出信息
  :return: 特征sample
  '''
  sample_cache = '{}-{}-{}'.format(model_name, pick_layer, dataset)  # 保存特征文件名

  try:
    samples = cPickle.load(open(os.path.join(feature_path, sample_cache), "rb", True))
    if verbose:
      print("Using features...%s," % (sample_cache))
  except:
    if verbose:
      print("Extract features..., ")

    feat_model = Model(inputs=model.input, outputs=model.get_layer(pick_layer).output)
    # TO SECA_c
    # feat_model=load_model(model, custom_objects={'<lambda>': lambda y_true, y_pred: y_pred})
    # feat_model = Model(inputs=feat_model.get_layer('input_1').input, outputs=feat_model.get_layer(pick_layer).output)
    # # 编译
    # lambda_c = 0.002
    # feat_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # print(feat_model.summary())

    samples = []
    data = db.get_data()
    temp = 0
    for d in data.itertuples():
      d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
      '''
      相应的图像预处理 
      '''
      # TODO
      img = image.load_img(d_img, target_size=(image_size, image_size), color_mode=color_mode)
      img = image.img_to_array(img)
      img = img / 255.
      img = np.expand_dims(img, axis=0)

      try:
        d_hist = feat_model.predict(img)
        d_hist = np.sum(d_hist, axis=0)
        d_hist /= np.sum(d_hist)  # normalize
        samples.append({
          'img': d_img,
          'cls': d_cls,
          'hist': d_hist,
          'cns': db.get_class_nums(d_cls)
        })
        temp += 1
        if verbose:
          print(d_img, '###', temp)
      except:
        # pass
        print("!!!!Wrong!!!!", d_img)
    cPickle.dump(samples, open(os.path.join(feature_path, sample_cache), "wb", True))
  return samples


def single_query_infer(q, s_samples, depth=10, d_type=1):
  '''
  :param q_sample: 查询图像
  :param s_samples:检索库
  :param depth: 检索深度
  :param d_type: 检索距离测度
  :return: ap,ar,P11 和 检索排序结果result
  '''


  q_hist = q
  results = []
  d_types=['cosine', 'd2', 'd1', 'hausdorff']
  print('use distance:', d_types[d_type])
  for idx, sample in enumerate(s_samples):
    s_img, s_cls, s_hist, s_cns = sample['img'], sample['cls'], sample['hist'], sample['cns']

    results.append({
                    'dis': distance(q_hist, s_hist, d_type=d_types[d_type]),
                    'cls': s_cls,
                    'img': s_img,
                    'cns': s_cns
                  })
  results = sorted(results, key=lambda x: x['dis']) # 结果按距离排序

  if depth and depth <= len(results): # 返回深度
    results = results[:depth]

  results_path = [r['img'] for r in results]

  return results_path

