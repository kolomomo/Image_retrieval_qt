from skimage.draw import circle
import numpy as np
import skimage.io
import math
from skimage.viewer import ImageViewer
import csv
import time
import os

def calcRadius(shape):
	val = float(min(shape[0],shape[1]))/2
	radii = [None]*5
	radii[4] = val
	for i in range(0,4):
		radii[i] = (float(math.sqrt(i+1))*radii[4])/math.sqrt(5)
	return radii


directory = '/home/wbo/PycharmProjects/Image_retrieval_qt/CBIR_Dict/subset'



images = os.listdir(directory)
csvfile = open('/home/wbo/PycharmProjects/Image_retrieval_qt/CBIR_Dict/feature/features32.csv', 'wb')
writer = csv.writer(csvfile, delimiter=',')

for imgname in images:
	print(imgname)
	img = skimage.io.imread(directory+'/'+imgname)
	x,y = np.shape(img)
	center = (x/2,y/2)
	subimg = [None]*5
	subimg[0] = np.array(img[0:center[0],0:center[1]])
	subimg[1] = img[0:center[0],center[1]:y]
	subimg[2] = img[center[0]:x,0:center[1]]
	subimg[3] = img[center[0]:x,center[1]:y]
	xl = int(center[0]-float(x)/4)
	xr = int(center[0]+float(x)/4)
	yu = int(center[1]-float(y)/4)
	yd = int(center[1]+float(y)/4)
	subimg[4] = img[xl:xr,yu:yd]
	shape = np.shape(subimg[0])
	radii = calcRadius(shape)
	subimx = float(shape[0])/2.0
	subimy = float(shape[1])/2.0
	points = []
	for i in range(0,5):
		points.append([])

	for k in range(0,5):
		rr,cc = circle(subimx, subimy, radii[k])
		for i,j in zip(rr,cc):
			points[k].append((i,j))

	for i in range(4,0,-1):
		points[i] = [p for p in points[i] if p not in points[i-1]]


	k=1
	feat = [None]*41
	feat[0]=imgname
	for im in subimg:
		for pt in points[1:5]:
			m=0
			std = 0
			for t in pt:
				m = m + im[t[0]][t[1]]
				std = std + (im[t[0]][t[1]])**2
			feat[k] = float(m)/len(pt)
			feat[k+1] = std - len(pt)*(feat[k]**2)
			k = k + 2


	writer.writerow(feat)

csvfile.close()
