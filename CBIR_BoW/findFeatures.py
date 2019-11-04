import cv2
import numpy as np
import os
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn import preprocessing

train_path = "/home/wbo/Datasets/ir_train"
numWords = 1000

training_names = os.listdir(train_path)

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
for training_name in training_names:
    class_path = os.path.join(train_path, training_name)
    for im in os.listdir(class_path):
        image_path = os.path.join(class_path, im)
        image_paths += [image_path]

# Create feature extraction and keypoint detector objects
detector = cv2.xfeatures2d.SIFT_create()

# List where all the descriptors are stored
des_list = []
remove_list = []
for i, image_path in enumerate(image_paths):
    im = cv2.imread(image_path)
    # print("Extract SIFT of %s image, %d of %d images" %(image_path, i, len(image_paths)))
    kpts, des = detector.detectAndCompute(im, None)
    if des is not None:
        des_list.append((image_path, des))
    else:
        print('000000000', image_path)
        remove_list.append(image_path)

for r in remove_list:
    image_paths.remove(r)
    print('remove succes, new length:', len(image_paths))

# Stack all the descriptors vertically in a numpy array
print(len(des_list))
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))  

# Perform k-means clustering
print("Start k-means: %d words, %d key points" % (numWords, descriptors.shape[0]))
voc, variance = kmeans(descriptors, numWords, 1)

# Calculate the histogram of features
im_features = np.zeros((len(image_paths), numWords), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Perform L2 normalization
im_features = im_features*idf
im_features = preprocessing.normalize(im_features, norm='l2')

joblib.dump((im_features, image_paths, idf, numWords, voc), "bof.pkl", compress=3)
