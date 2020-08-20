# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   FileName:    grad_camv2
   Author:      kolomomo
   Date:        2020/8/13
   UpdateData:  2020/8/13:
-------------------------------------------------
   Description: 热力图
   
-------------------------------------------------
"""

from keras import backend as K
import numpy as np
import cv2


def heatmap(model,
            data_img,
            layer_idx,
            img_show=None,
            pred_idx=None):

    if pred_idx is None:
        # 预测
        _, preds = model.predict(data_img)
        # 获取最高预测项的index
        pred_idx = np.argmax(preds[0])
    # 目标输出估值
    target_output = model.output[:, pred_idx]
    # 目标层的输出代表各通道关注的位置
    last_conv_layer_output = model.layers[layer_idx].output
    # 求最终输出对目标层输出的导数(优化目标层输出),代表目标层输出对结果的影响
    grads = K.gradients(target_output, last_conv_layer_output)[0]
    # 将每个通道的导数取平均,值越高代表该通道影响越大
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer_output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([data_img])
    # 将各通道关注的位置和各通道的影响乘起来
    for i in range(conv_layer_output_value.shape[-1]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    # 对各通道取平均得图片位置对结果的影响
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    # 规范化
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    # plt.matshow(heatmap)
    # plt.show()
    # 叠加图片
    # 缩放成同等大小
    heatmap = cv2.resize(heatmap, (img_show.shape[1], img_show.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    # 将热图应用于原始图像.由于opencv热度图为BGR,需要转RGB
    superimposed_img = img_show + cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)[:,:,::-1] * 0.4
    # 截取转uint8
    superimposed_img = np.minimum(superimposed_img, 255).astype('uint8')
    return superimposed_img, heatmap

# 生成所有卷积层的热度图
# def heatmaps(model, data_img, img_show=None):
#     if img_show is None:
#         img_show = np.array(data_img)
#     # Resize
#     input_shape = K.int_shape(model.input)[1:3]  # (28,28,1)
#     data_img = image.img_to_array(image.array_to_img(data_img).resize(input_shape))
#     # 添加一个维度->(1, 224, 224, 3)
#     data_img = np.expand_dims(data_img, axis=0)
#     # 预测
#     preds = model.predict(data_img)
#     # 获取最高预测项的index
#     pred_idx = np.argmax(preds[0])
#     print("预测为:%d(%f)" % (pred_idx, preds[0][pred_idx]))
#     indexs = []
#     for i in range(model.layers.__len__()):
#         if 'conv' in model.layers[i].name:
#             indexs.append(i)
#     print('模型共有%d个卷积层' % indexs.__len__())
#     # plt.suptitle('heatmaps for each conv')
#     # for i in range(indexs.__len__()):
#     #     ret = heatmap(model, data_img, indexs[i], img_show=img_show, pred_idx=pred_idx)
#     #     plt.subplot(np.ceil(np.sqrt(indexs.__len__()*2)), np.ceil(np.sqrt(indexs.__len__()*2)), i*2 + 1)\
#     #         .set_title(model.layers[indexs[i]].name)
#     #     plt.imshow(ret[0])
#     #     plt.axis('off')
#     #     plt.subplot(np.ceil(np.sqrt(indexs.__len__()*2)), np.ceil(np.sqrt(indexs.__len__()*2)), i*2 + 2)\
#     #         .set_title(model.layers[indexs[i]].name)
#     #     plt.imshow(ret[1])
#     #     plt.axis('off')
#     # plt.show()


def make_heatmap(x,
                 model,
                 preds,
                 layer):

    index = np.argmax(preds[0])
    model_output = model.output[1][:, index]

    layer_name = layer  # choose last layer of conv
    layer_output = model.get_layer(layer_name).output

    grads = K.gradients(model_output, layer_output)[0]  # ∂model_output/∂layer_output
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    my_function = K.function([model.input], [pooled_grads, layer_output[0]])

    pooled_grads_value, layer_output_value = my_function([x])  # cast an image

    for i in range(512):
        layer_output_value[:, :, i] = layer_output_value[:, :, i] * pooled_grads_value[i]

    heatmap = np.mean(layer_output_value, axis=-1)
    # print('heatmap.shape',heatmap.shape)#(14,14)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    # plt.matshow(heatmap)#matshow for a matrix
    # plt.show()
    return heatmap


def show_heatmap(img_path,
                 heatmap):

    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmapshow = None
    heatmapshow = cv2.normalize(heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)

    # cv2.imshow('heatmapshow', heatmapshow)
    # cv2.imshow('original image', img)
    # print('heatmapshow.shape',heatmapshow.shape)
    # print('original image',img.shape)
    assert heatmapshow.shape == img.shape, 'shape does not match'

    cv2.namedWindow('Heatmap Image', cv2.WINDOW_NORMAL)  # change the size of image in GU

    def nothing(x):
        pass

    cv2.createTrackbar('blending[%]', 'Heatmap Image', 0, 100, nothing)
    while 1:
        # alpha blending
        # https://teratail.com/questions/190326
        alpha = cv2.getTrackbarPos('blending[%]', 'Heatmap Image')
        blended = cv2.addWeighted(heatmapshow, alpha / 100, img, 1 - alpha / 100, 0)
        cv2.imshow('Heatmap Image', blended)

        k = cv2.waitKey(1)  # https://qiita.com/MuAuan/items/d7d0202cd29c92f76bf4
        # if this is cv2.waitKey(0), while loop doesn't go round
        # because waitKey(0) means to wait permanently until any key is clicked
        if k == 27:  # if you click [esc] key, while loop would break
            cv2.destroyAllWindows()
            break
    return 0


if __name__ == '__main__':
    pass