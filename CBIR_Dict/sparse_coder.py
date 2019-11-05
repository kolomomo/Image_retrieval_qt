from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import itertools
import csv
import math
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.decomposition import DictionaryLearning
from sklearn.preprocessing import normalize
from sklearn.decomposition import SparseCoder
import copy
import os
import skimage.io
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean, cosine

# np.set_printoptions(threshold=np.nan)

class cluster(object):
    def toFloat(self):
        for i in range(0, len(self.dataset)):
            for j in range(0, self.columns):
                # print(i,j, self.dataset[i][j])
                self.dataset[i][j] = float(self.dataset[i][j])

    def toFloatn(self):
        for i in range(0, len(self.datasetn)):
            for j in range(0, self.columnsn): # 1 TODO
                self.datasetn[i][j] = float(self.datasetn[i][j])

    def __init__(self):
        f = open('/home/wbo/PycharmProjects/Image_retrieval_qt/CBIR_Dict/feature/features_S.csv', 'r')
        d = ','
        reader1, dataset = itertools.tee(csv.reader(f, delimiter=d))
        Q_csv = '/home/wbo/PycharmProjects/Image_retrieval_qt/datasets/ir_test.csv'
        Q_images = pd.read_csv(Q_csv)
        S_csv = '/home/wbo/PycharmProjects/Image_retrieval_qt/datasets/ir_train.csv'
        S_images = pd.read_csv(S_csv)
        self.Q = [image for image, label in Q_images.values]
        self.S = [image for image, label in S_images.values]


        self.columns = len(next(reader1))
        del reader1
        self.dataset = list(dataset)
        self.rows = len(self.dataset)
        self.toFloat()
        self.datasetcopy = copy.deepcopy(self.dataset)
        self.dataset = self.normalizeDS(self.dataset, self.rows, self.columns)
        self.dataset = self.sknormalize(self.dataset)
        self.n_components = 60
        self.n_clusters = 4 # TODO 4
        self.n_nonzero_coefs = 10
        f.close()

        f2 = open('/home/wbo/PycharmProjects/Image_retrieval_qt/CBIR_Dict/feature/features_Q.csv', 'r')
        readern, datasetn = itertools.tee(csv.reader(f2, delimiter=d))
        self.columnsn = len(next(readern))
        del readern
        self.datasetn = list(datasetn)
        self.rowsn = len(self.datasetn)
        self.toFloatn()
        f2.close()

        # f3 = open('name_class.csv', 'r')
        # d = ','
        # reader1, imnames = itertools.tee(csv.reader(f3, delimiter=d))
        # self.columnsimn = len(next(reader1))
        # del reader1
        # self.imnames = list(imnames)
        # self.imnames = np.array(self.imnames)
        # f3.close()

        self.imnames = [[image, label] for image, label in S_images.values]
        self.imnames = np.array(self.imnames)


    def normalizeDS(self, X, rows, columns):
        dsMax = np.amax(X, axis=0)
        dsMin = np.amin(X, axis=0)
        for i in range(0, rows):
            for j in range(0, columns):
                X[i][j] = float(X[i][j] - dsMin[j]) / (dsMax[j] - dsMin[j])
        return X

    def sknormalize(self, X):
        X = normalize(np.array(X))
        return X

    def kmcluster(self):
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(self.dataset)
        labels = self.kmeans.labels_
        y = np.bincount(self.kmeans.labels_)
        ii = np.nonzero(y)[0]

        for ii in range(self.n_clusters):
            print(ii, y[ii])
        self.cluster = []
        self.clustern = []

        for i in range(self.n_clusters):
            self.cluster.append([])
            self.clustern.append([])

        for l in range(len(labels)):
            self.cluster[labels[l]].append(list(self.dataset[l]))
            self.clustern[labels[l]].append(l)

        for i in range(self.n_clusters):
            self.dataset[i] = np.array(self.dataset[i])

        for i in range(self.n_clusters):
            self.cluster[i] = np.array(self.cluster[i])

    def DL(self):
        dictLearn = DictionaryLearning(n_components=self.n_components, alpha=1, transform_algorithm='omp',
                                       transform_n_nonzero_coefs=20)
        self.dictionary = []
        gamma = []
        # print self.cluster[0]
        # print self.cluster
        for i in range(self.n_clusters):
            print(self.cluster[i])
            dictObject = dictLearn.fit(self.cluster[i])
            self.dictionary.append(dictObject.components_)
            gamma.append(dictObject.transform(self.cluster[i]))

    def concatDict(self):
        self.concatD = []
        for i in self.dictionary:
            for j in i:
                self.concatD.append(j)
        self.concatD = np.array(self.concatD)
        self.concatD = self.normalizeDS(self.concatD, np.shape(self.concatD)[0], np.shape(self.concatD)[1])
        self.concatD = self.sknormalize(self.concatD)

    def sparseCode(self):
        self.omp = SparseCoder(self.concatD, transform_algorithm='omp', transform_n_nonzero_coefs=self.n_nonzero_coefs)

    def process(self):
        itr = 0
        thresh = 99999
        prev_sizes = None
        while itr < 5:
            itr = itr + 1
            curr_sizes = []
            self.cluster = [] * self.n_clusters
            self.clustern = [] * self.n_clusters
            for i in range(self.n_clusters):
                self.cluster.append([])
                self.clustern.append([])
            # print self.cluster

            for i in range(0, len(self.dataset)):
                sparse = self.omp.transform(np.array(self.dataset[i]).reshape(1, -1))
                k = 0
                errors = []
                for j in range(self.n_clusters):
                    sparseT = np.array(sparse[0][k:k + self.n_components])
                    dicti = self.dictionary[j]
                    res = np.dot(sparseT, dicti)
                    errors.append(math.sqrt(sum((res - self.dataset[i]) ** 2)))
                    k = k + self.n_components
                # t = input()
                ind = errors.index(min(errors))
                self.cluster[ind].append(list(self.dataset[i]))
                self.clustern[ind].append(i)
            # print self.cluster[0]
            print(itr)
            for cl in range(self.n_clusters):
                self.cluster[cl] = np.array(self.cluster[cl])
                curr_sizes.append(np.shape(self.cluster[cl])[0])
            if prev_sizes is not None:
                thresh = np.sum(abs(np.array(curr_sizes) - np.array(prev_sizes)))
            prev_sizes = copy.deepcopy(curr_sizes)
            print(thresh)
            self.DL()
            self.concatDict()
            self.sparseCode()
        # print self.cluster[0][0]
        # t=input()

    def query(self, q=2):
        q = q
        # qftrs = self.datasetn[q][1:41]
        qftrs = self.datasetn[q]
        k = 0
        errors = []
        qsparse = self.omp.transform(np.array(qftrs).reshape(1, -1))
        for j in range(self.n_clusters):
            sparseT = np.array(qsparse[0][k:k + self.n_components])
            dicti = self.dictionary[j]
            res = np.dot(sparseT, dicti)
            errors.append(math.sqrt(sum((res - qftrs) ** 2)))
            k = k + self.n_components
        ind = errors.index(min(errors))
        # print("query = ", self.datasetn[q][0])
        print("query = ", self.Q[q])
        # dir1 = '/media/aparna/C6A2E75FA2E7530B/Users/admin/Documents/subset'
        # for i in self.clustern[ind]:
        #     print(self.datasetn[i][0])
        fig = plt.figure()
        img1 = skimage.io.imread(self.Q[q-1])
        a = fig.add_subplot(5, 3, 1)
        imgplot = plt.imshow(img1)
        ed = []
        for i in range(len(self.cluster[ind])):
            ed.append((self.S[i], cosine(self.cluster[ind][i], self.dataset[q])))
        self.des = sorted(ed, key=lambda x: x[1])[:10] # 返回前十

        for i in range(2, 12):
            a = fig.add_subplot(5, 3, i)
            img1 = skimage.io.imread(self.des[i - 2][0])
            imgplot = plt.imshow(img1)
        plt.show()

        # precision
        classq = self.imnames[q - 1][1]
        print('predict label:', classq)
        corr = 0
        print("class of query=", classq, "image name=", self.imnames[q - 1][0])
        for i in range(0, len(self.des)):
            print(self.des[i][0], self.imnames[np.where(self.imnames[:, 0] == self.des[i][0])[0][0]][1])
            # cli = self.imnames[np.where(self.imnames[:,0] == ((self.des[i][0].split('.')[0]).split(' ')[0]))[0]][1]
            # print cli
            clret = self.imnames[np.where(self.imnames[:, 0] == self.des[i][0])[0][0]][1]
            print('result label:', clret)
            if clret == classq:
                corr = corr + 1
        precision = float(corr) / 10.0 # 前十
        print('precision', precision)

    def queryKm(self, q=2):
        q = q
        qftrs = self.datasetn[q]
        centers = self.kmeans.cluster_centers_
        edk = []
        for j in range(len(centers)):
            edk.append(euclidean(qftrs, centers[j]))
        ind = edk.index(min(edk))
        ed = []
        # dir1 = '/media/aparna/C6A2E75FA2E7530B/Users/admin/Documents/subset'

        for i in range(0, len(self.cluster[ind])):
            ed.append((self.S[i], cosine(self.cluster[ind][i], self.dataset[q])))
        self.des = sorted(ed, key=lambda x: x[1])[:10]
        fig = plt.figure()
        img1 = skimage.io.imread(self.Q[q-1])
        a = fig.add_subplot(5, 3, 1)
        imgplot = plt.imshow(img1)
        print('len(self.des):',len(self.des))

        for i in range(2, 12):
            a = fig.add_subplot(5, 3, i)
            img1 = skimage.io.imread(self.des[i - 2][0])
            imgplot = plt.imshow(img1)
        plt.show()

        # precision
        classq = self.imnames[q - 1][1]
        print("class of query=", classq, "image name=", self.imnames[q - 1][0])
        corr = 0
        for i in range(0, len(self.des)):
            clret = self.imnames[np.where(self.imnames[:, 0] == self.des[i][0])[0][0]][1]
            print(clret)
            if clret == classq:
                corr = corr + 1
        precision = float(corr) / 10.0
        print(precision)


clusterObj = cluster()
clusterObj.kmcluster()
clusterObj.DL()
clusterObj.concatDict()
clusterObj.sparseCode()
clusterObj.process()
clusterObj.query(q=3)
clusterObj.queryKm(q=3)
