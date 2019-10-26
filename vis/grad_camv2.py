# from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras import backend as K
import numpy as np
import cv2

# import matplotlib.pyplot as plt

def make_heatmap(x, model, preds, layer):
    # print('preds.shape',preds.shape)#[[1000 classes prediction]](1,1000)
    # print('preds[0]',preds[0])#[1000 classes prediction]

    index = np.argmax(preds[0])
    # print('index:',index)#index:0~1000
    # print('preds[0][index]',preds[0][index])
    # print('index:', index)
    # print(decode_predictions(preds)[0])  # decode results of prediction so that you can read it

    model_output = model.output[1][:, index]
    # print('model_output1',model_output)

    layer_name = layer  # choose last layer of conv
    layer_output = model.get_layer(layer_name).output
    # print('layer_output',layer_output)#Tensor("block5_conv3/Relu", shape=(?, 14, 14, 512))

    grads = K.gradients(model_output, layer_output)[0]  # ∂model_output/∂layer_output
    # https://keras.io/backend/#backend-functions
    # K.gradients(loss, variables)
    # loss: Scalar tensor to minimize, variables: List of variables.
    # K.gradidents returns a gradients tensor
    # print('grads', grads)#Tensor("gradients", shape=(?, 14, 14, 512))

    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    # print('pooled_grads',pooled_grads)#(512,)

    my_function = K.function([model.input], [pooled_grads, layer_output[0]])
    # layer_output[0], shape=(14,14,512)
    # K.function(inputs, outputs), create function as you like

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


def show_heatmap(img_path, heatmap):
    img = cv2.imread(img_path)  # use cv2 to load the original image

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # resize the heatmap to have the same size as the original image

    heatmapshow = None  # https://stackoverflow.com/questions/56275515/visualizing-a-heatmap-matrix-on-to-an-image-in-opencv
    heatmapshow = cv2.normalize(heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # apply the heatmap to the original image
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)

    # cv2.imshow('heatmapshow', heatmapshow)
    # cv2.imshow('original image', img)
    # print('heatmapshow.shape',heatmapshow.shape)
    # print('original image',img.shape)
    assert heatmapshow.shape == img.shape, 'shape does not match'

    cv2.namedWindow('Heatmap Image', cv2.WINDOW_NORMAL)  # change the size of image in GUI

    # cv2.namedWindow("elephant")

    def nothing(x):
        pass

    cv2.createTrackbar('blending[%]', 'Heatmap Image', 0, 100, nothing)
    while (1):
        # alpha blending
        # https://teratail.com/questions/190326
        alpha = cv2.getTrackbarPos('blending[%]', 'Heatmap Image')
        blended = cv2.addWeighted(heatmapshow, alpha / 100, img, 1 - alpha / 100, 0)
        cv2.imshow('Heatmap Image', blended)

        k = cv2.waitKey(1)  # https://qiita.com/MuAuan/items/d7d0202cd29c92f76bf4
        # if this is cv2.waitKey(0), while loop doesn't go round
        # because waitKey(0) means to wait permanently until any key is clicked
        if k == 27:  # if you click [esc] key, while loop would break
            break
            cv2.destroyAllWindows()
    return 0