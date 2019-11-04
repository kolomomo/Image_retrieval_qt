import cv2
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn import preprocessing
import numpy as np

from pylab import *
from PIL import Image
from util.data import Database

S_PATH = '/home/wbo/Datasets/ir_train_S' # 检索数据集
save_S_csv = '/home/wbo/PycharmProjects/Image_retrieval_qt/datasets/ir_train.csv'
sdb = Database(data_path=S_PATH, save_path=save_S_csv)

image_path = '/home/wbo/Datasets/ir_val_Q/1121-200-411-700/2229.jpg'
# Load the classifier, class names, scaler, number of clusters and vocabulary
im_features, image_paths, idf, numWords, voc = joblib.load("bof.pkl")

# Create feature extraction and keypoint detector objects
detector = cv2.xfeatures2d.SIFT_create()
# fea_det = cv2.FeatureDetector_create("SIFT")
# des_ext = cv2.DescriptorExtractor_create("SIFT")

# List where all the descriptors are stored
des_list = []

im = cv2.imread(image_path)
im = cv2.resize(im, (128, 128))
kpts, des = detector.detectAndCompute(im, None)
# kpts = fea_det.detect(im)
# kpts, des = des_ext.compute(im, kpts)

# rootsift
# rs = RootSIFT()
# des = rs.compute(kpts, des)

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

# Visualize the results
figure()
gray()
subplot(5, 5, 1)
imshow(im[:, :, ::-1])
axis('off')
for i, ID in enumerate(rank_ID[0][0:20]):
    img = Image.open(image_paths[ID])
    gray()
    subplot(5, 5, i + 6)
    imshow(img)
    axis('off')
    # print(image_paths[ID])
show()




#
# print('ap, ar:', ap, ar)
# print('P11 mean:', np.mean(P11, axis=0))
# np.save('/home/wbo/PycharmProjects/Image_retrieval_qt/features/P11/{}_P11'.format(model_name), np.mean(P11, axis=0))

