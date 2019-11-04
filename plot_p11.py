import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

CenterVLAD_P11 = np.load('/home/wbo/PycharmProjects/Image_retrieval_qt/features/P11/CenterVLAD_P11.npy')
ResNet50_P11 = np.load('/home/wbo/PycharmProjects/Image_retrieval_qt/features/P11/ResNet50_P11.npy')
SECA_P11 = np.load('/home/wbo/PycharmProjects/Image_retrieval_qt/features/P11/SECA_P11.npy')
VGG16_P11 = np.load('/home/wbo/PycharmProjects/Image_retrieval_qt/features/P11/VGG16_P11.npy')
ResNet50_CenterVLAD_P11 = np.load('/home/wbo/PycharmProjects/Image_retrieval_qt/features/P11/ResNet50_CenterVLAD_P11.npy')

xx = [i/10 for i in range(11)]
# mpl.rcParams['font.family'] = ['simhei']
plt.figure(figsize=(15,10))

plt.xlabel('Recall', fontsize=16)
plt.ylabel('mAP', fontsize=16)

plt.plot(xx, ResNet50_CenterVLAD_P11, marker='h', label='ResNet50_CenterVLAD', markersize=8)
plt.plot(xx, CenterVLAD_P11, marker='^', label='CenterVLAD', markersize=8)
plt.plot(xx, VGG16_P11, marker='d',label='VGG16', markersize=8)
plt.plot(xx, SECA_P11, marker='D',label='SECA', markersize=8)
plt.plot(xx, ResNet50_P11, marker='H',label='ResNet50', markersize=8)

# plt.ylim(0.1, 1)
plt.legend(loc='best', fontsize='x-large')
# plt.savefig('./p11_mAP_Recall_in')
plt.show()