import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

CenterVLAD_P11 = np.load('/home/wbo/PycharmProjects/Image_retrieval_qt/features/P11/CenterVLAD_P11.npy')
ResNet50_P11 = np.load('/home/wbo/PycharmProjects/Image_retrieval_qt/features/P11/ResNet50_P11.npy')
SECA_P11 = np.load('/home/wbo/PycharmProjects/Image_retrieval_qt/features/P11/SECA_P11.npy')
VGG16_P11 = np.load('/home/wbo/PycharmProjects/Image_retrieval_qt/features/P11/VGG16_P11.npy')
ResNet50_CenterVLAD_P11 = np.load('/home/wbo/PycharmProjects/Image_retrieval_qt/features/P11/ResNet50_CenterVLAD_P11.npy')
Tamura = np.load('/home/wbo/Project/SEA/retrieval/P11/Tamura_P11.npy')
CNNs = np.load('/home/wbo/PycharmProjects/Image_retrieval_qt/features/P11/CNNs_P11.npy')
IFCFW = np.load('/home/wbo/PycharmProjects/Image_retrieval_qt/features/P11/AttentionResNet_P11.npy')

Baseline = [0.44,0.41,0.39,0.38,0.35,0.36,0.18,0.17,0.18,0.17,0.05]
xx = [i/10 for i in range(11)]
# mpl.rcParams['font.family'] = ['simhei']
plt.figure(figsize=(15,10))

plt.xlabel('Recall', fontsize=24)
plt.ylabel('mAP', fontsize=24)
plt.tick_params(labelsize=16)

# plt.plot(xx, CenterVLAD_P11, marker='^', label='SEA+CenterVLAD', markersize=8)
plt.plot(xx, ResNet50_CenterVLAD_P11, marker='h', label='ResNet50_CenterVLAD', markersize=8)
# plt.plot(xx, VGG16_P11, marker='d',label='VGG16', markersize=8)
plt.plot(xx, SECA_P11, marker='D',label='SECA', markersize=8)
plt.plot(xx, ResNet50_P11, marker='H',label='ResNet50', markersize=8)
# plt.plot(xx, IFCFW, marker='d',label='IF+CFW', markersize=8)
#
# plt.plot(xx, Baseline, marker='H',label='Baseline', markersize=8)
# plt.plot(xx, CNNs, marker='*',label='CNNs', markersize=8)
# plt.plot(xx, Tamura, marker='s',label='Tamura', markersize=8)


# plt.ylim(0.1, 1)
plt.legend(loc='best', fontsize='x-large')
# plt.savefig('./p11_mAP_Recall_in')
plt.show()