import numpy as np

def eva_ap_ar(r,t):
    '''
    计算ap,ar
    r: 预测结果标签
    t：真实标签
    '''
    classes = list(set(t))
#     N = len(classes)
    APs = []
    ARs = []
#     TN = []
    for c in classes:
        tc = set(np.where(t==c)[0])
        pc = set(np.where(r==c)[0])
        TPc = len(pc & tc)
        FPc = len(pc - tc)
        if (TPc + FPc):
            AP = TPc / (TPc + FPc)
        else:
            AP = 0.0
        AR = TPc / len(tc)
        APs.append(AP)
        ARs.append(AR)
#         print('AP:{0:.4f},  AR:{1:.4f}'.format(AP, AR))
    return APs, ARs

def AP(label, results, sort=True, P11=True):
  precision = []
  recall = []
  hit = 0
  for i, result in enumerate(results):
    if result == label:
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