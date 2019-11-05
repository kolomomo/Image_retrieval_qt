import cv2
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn import preprocessing
import numpy as np
from pylab import *
from PIL import Image
from util.data import Database
import pandas as pd


def AP(label, results, P11=True):
    precision = []
    recall = []
    hit = 0
    for i, result in enumerate(results):
        if result == label:
            hit += 1
            precision.append(hit / (i + 1.))
            recall.append(hit / sdb.get_class_nums(label))

    if hit == 0:
        if P11:
            return 0., 0., [0. for i in range(11)]
        else:
            return 0., 0.

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


def searchF(q_image, q_label):
    image_path = q_image
    # Load the classifier, class names, scaler, number of clusters and vocabulary
    im_features, image_paths, idf, numWords, voc = joblib.load("bof.pkl")

    # Create feature extraction and keypoint detector objects
    detector = cv2.xfeatures2d.SIFT_create()

    # List where all the descriptors are stored
    des_list = []
    im = cv2.imread(image_path)
    im = cv2.resize(im, (128, 128))
    kpts, des = detector.detectAndCompute(im, None)
    if des is not None:
        des_list.append((image_path, des))

        # Stack all the descriptors vertically in a numpy array
        descriptors = des_list[0][1]

        #
        test_features = np.zeros((1, numWords), "float32")
        words, distance = vq(descriptors, voc)
        for w in words:
            test_features[0][w] += 1

        # Perform Tf-Idf vectorization and L2 normalization
        test_features = test_features * idf
        test_features = preprocessing.normalize(test_features, norm='l2')

        score = np.dot(test_features, im_features.T)
        rank_ID = np.argsort(-score)

        r_label = []
        for i, ID in enumerate(rank_ID[0]):
            r_label.append(image_paths[ID].split('/')[-2])

        precision, recall, P = AP(q_label, r_label)
        ap = np.mean(precision)
        ar = np.mean(recall)

        return ap, ar, P
    else:
        return None

Q_PAYH = '/home/wbo/Datasets/ir_val_Q' # 待检索数据集
S_PATH = '/home/wbo/Datasets/ir_train_S' # 检索数据集
save_Q_csv = '/home/wbo/PycharmProjects/Image_retrieval_qt/datasets/ir_test.csv'
save_S_csv = '/home/wbo/PycharmProjects/Image_retrieval_qt/datasets/ir_train.csv'
#
sdb = Database(data_path=S_PATH, save_path=save_S_csv)
#
qcsv = pd.read_csv(save_Q_csv)
Q = [[im,label] for im,label in qcsv.values]


all_ap = []   # 所有查询ap值
P11 = []
for query in Q:
    print(query)
    result = searchF(query[0], query[1])
    if result is not None:
        ap, ar, p11 = result
        all_ap.append(ap)
        P11.append(p11)
    else:
        pass

print("all MAP", np.mean(all_ap))
print('P11 mean:', np.mean(P11, axis=0))





