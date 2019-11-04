import os
from PIL import Image
import numpy as np
import scipy.io as sio

def combine_image(fd_list):
    '''
    按通道重叠不同图像，保存为numpy格式
    :param fd_list: 图像完整路径列表
    :return:
    '''
    img = Image.open(fd_list[0])
    img = np.array(img)
    for fn in fd_list[1:]:
        im = Image.open(fn)
        im = np.array(im)
        img = np.dstack((img,im))
    return img

def combine_dir(dir_list, save_dir):
    '''

    :param dir_list:
    :param save_dir:
    :return:
    '''
    list_all = []
    for d in dir_list:
        list1 = [list1 for list1,_,_ in os.walk(d)][1:]
        if not len(list_all):
            list_all = list1
            print(len(list_all))
            z = list_all
        else:
            z = zip(list_all, list1)

    for zz in list(z):
        c = os.path.join(save_dir, zz[0].split('/')[-1])
        if not os.path.exists(c):
            os.mkdir(c)
        for zzz in os.listdir(zz[0]):
            ll=[]
            n = zzz.split('.')[0]+'.mat'
            for i in range(len(zz)):
                ll.append(os.path.join(zz[i], zzz))
            save_fn = os.path.join(c, n)
            img = combine_image(ll)
            sio.savemat(save_fn, {'data': img})
    return

dir_list = ['/home/wbo/Project/SEA/datasets/ir_val1','/home/wbo/Project/SEA/datasets/ir_val2']
save_d = '/home/wbo/Project/SEA/datasets/ir_val3'
if not os.path.exists(save_d):
    os.mkdir(save_d)
combine_dir(dir_list, save_dir=save_d)
# im_list = ['/home/wbo/Project/SEA/datasets/ir_train_rescale_128/1121-4a0-310-700/3770.jpg',
#            '/home/wbo/Project/SEA/datasets/ir_train_rescale_128/1121-4a0-310-700/3770.jpg']
# im = combine_image(im_list)
# sio.savemat('./1.mat', {'data':im})
# img = sio.loadmat('./1.mat')
#
# print(img['data'].shape)
