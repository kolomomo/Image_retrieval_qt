from retrieval.ir_query_v2 import *
from util.data import Database
import matplotlib.pyplot as plt
from PIL import Image

S_path = '/home/wbo/PycharmProjects/Image_retrieval_qt/features/CenterVLAD-feature-SData'
Q_path = '/home/wbo/PycharmProjects/Image_retrieval_qt/features/CenterVLAD-feature-QData'

d_type = 'cosine'

S = cPickle.load(open(S_path, "rb", True))
Q = cPickle.load(open(Q_path, "rb", True))

ap, ar, p11, r = query_infer(Q[33], s_samples=S, depth=20, d_type=d_type)
print(ap)
# fig, axes = plt.subplots(5, 5, figsize=(15, 15))

# k=0
# im = Image.open(Q[0]['img'])
# axes[0,0].imshow(im, cmap='gray')
# axes[0,0].set_title(Q[0]['cls'])
#
# for i in range(5):
#     axes[0, i].axis('off')
# for i in range(1, 5):
#     for j in range(5):
#         im = Image.open(r[k]['img'])
#         axes[i,j].imshow(im, cmap='gray')
#         # 为每个子图添加标题
#         axes[i,j].set_title(r[k]['cls'])
#         axes[i,j].axis('off')
#         k=k+1
# plt.savefig('./temp/CenterVLAD_q1.svg')
# plt.show()

# Visualize the results
from pylab import *

figure()
gray()
subplot(5, 5, 1)
im = Image.open(Q[33]['img'])
im = im.resize((128,128))
print(Q[33]['img'])
imshow(im)
axis('off')
for k in range(20):
    img = Image.open(r[k]['img'])
    gray()
    subplot(5,5,k+6)
    imshow(img)
    axis('off')

show()