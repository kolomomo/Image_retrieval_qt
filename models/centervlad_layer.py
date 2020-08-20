# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   FileName:    CenterVLAD_layer
   Author:      kolomomo
   Date:        2020/8/13
   UpdateData:  2020/8/13:
-------------------------------------------------
   Description: 自定义CenterVLAD层
   
-------------------------------------------------
"""


import tensorflow as tf
from keras import initializers, layers, backend
# from keras.layers import Conv2D, Flatten
# from tensorflow.keras import layers

class CenterVLAD(layers.Layer):

    def __init__(self,
                 num_clusters=116,
                 assign_weight_initializer=None,
                 skip_postnorm=False,
                 **kwargs):

        self.K = num_clusters
        self.assign_weight_initializer = assign_weight_initializer
        self.skip_postnorm = skip_postnorm
        self.outdim = 3712
        self.D = None
        self.C = None

        self.conv = layers.Conv2D(filters=self.K,
                           kernel_size=1,
                           strides=(1, 1),
                           use_bias=False,
                           padding='valid',
                           kernel_initializer='zeros')
        self.flatten = layers.Flatten()

        super(CenterVLAD, self).__init__(**kwargs)

    def build(self, input_shape):
        self.D = input_shape[-1]
        self.outdim = input_shape[-1] * self.K
        self.C = self.add_weight(name='cluster_centers',
                                 shape=(1, 1, 1, self.D, self.K),
                                 initializer='zeros',
                                 dtype='float32',
                                 trainable=True)

        self.conv.build(input_shape)
        super(CenterVLAD, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        s = self.conv(inputs)
        a = tf.nn.softmax(s)
        # Dims used hereafter: batch, H, W, desc_coeff, cluster
        # Move cluster assignment to corresponding dimension.x = Lambda(lambda a: l2_normalize(a,axis=-1))(x)    #28
        a = tf.expand_dims(a, -2)
        # VLAD core.
        # v = tf.math.square(tf.expand_dims(inputs,-1) -self.C)
        v = tf.expand_dims(inputs, -1) + self.C
        v = a * v
        v = tf.reduce_sum(v, axis=[1, 2])
        v = tf.transpose(v, perm=[0, 2, 1])

        if not self.skip_postnorm:
            # Result seems to be very sensitive to the normalization method
            # details, so sticking to matconvnet-style normalization here.
            v = self.matconvnetNormalize(v, 1e-12)
            v = tf.transpose(v, perm=[0, 2, 1])
            v = self.flatten(v)
            v = self.matconvnetNormalize(v, 1e-12)
            # v = self.matconvnetNormalize(tf.layers.flatten(v), 1e-12)
        return v

    def matconvnetNormalize(self,
                            inputs,
                            epsilon):
        return inputs / tf.sqrt(tf.reduce_sum(inputs ** 2, axis=-1, keepdims=True)
                                + epsilon)

    def compute_output_shape(self,
                             input_shape):
        return tuple([None, self.outdim])
