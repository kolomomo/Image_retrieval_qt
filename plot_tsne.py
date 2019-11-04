import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.decomposition import PCA
import numpy as np
from six.moves import cPickle

tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
pca = PCA(n_components=5)

samples = cPickle.load(open('/home/wbo/PycharmProjects/Image_retrieval_qt/features/CenterVLAD-feature-SData', "rb", True))

X = [s['hist'] for s in samples]
y = [s['cls'] for s in samples]

x_pca = pca.fit_transform(X)
x_tsne = tsne.fit_transform(x_pca)
x_min, x_max = x_tsne.min(0),x_tsne.max(0)
x_norm = (x_tsne-x_min) / (x_max - x_min)

c = []
a = list(set(y))
for i in y:
    c.append(a.index(i))

plt.figure(figsize=(10, 6))
plt.scatter(x_norm[:,0], x_norm[:,1],c=c, s=2, cmap=plt.get_cmap('jet'))

plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.text(0.5,-0.1,'λ=0',fontsize=16)
# plt.savefig('./save/irtsne0')
plt.show()
