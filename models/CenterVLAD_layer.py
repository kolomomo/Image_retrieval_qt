import tensorflow as tf
from keras import initializers, layers
from keras.layers import Conv2D

class CenterVLAD(layers.Layer):
    """Creates a CenterVLAD class.
    """
    def __init__(self, num_clusters=116, assign_weight_initializer=None,
            cluster_initializer=None, skip_postnorm=False, **kwargs):

        self.K = num_clusters
        self.assign_weight_initializer = assign_weight_initializer
        self.skip_postnorm = skip_postnorm
        self.outdim = 3712
        super(CenterVLAD, self).__init__(**kwargs)

    def build(self, input_shape):
        self.D = input_shape[-1]
        self.C = self.add_weight(name='cluster_centers',
                                    shape=(1,1,1,self.D,self.K),
                                    initializer='zeros',
                                    dtype='float32',
                                    trainable=True)

        self.conv = Conv2D(filters = self.K,kernel_size=1,strides = (1,1),
            use_bias=False, padding = 'valid',
            kernel_initializer='zeros')
        self.conv.build(input_shape)

        #might be necessary for older versions where the weights of conv are not automatically added to
        #trainable_weights of the super-layer
        #self._trainable_weights.append(self.conv.trainable_weights[0])
        super(CenterVLAD, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):

        s = self.conv(inputs)
        a = tf.nn.softmax(s)

        # Dims used hereafter: batch, H, W, desc_coeff, cluster
        # Move cluster assignment to corresponding dimension.x = Lambda(lambda a: l2_normalize(a,axis=-1))(x)    #28
        a = tf.expand_dims(a,-2)

        # VLAD core.
        # v = tf.math.square(tf.expand_dims(inputs,-1) -self.C)
        v = tf.expand_dims(inputs,-1)+self.C
        v = a*v
        v = tf.reduce_sum(v,axis=[1,2])
        v = tf.transpose(v,perm=[0,2,1])

        if not self.skip_postnorm:
            # Result seems to be very sensitive to the normalization method
            # details, so sticking to matconvnet-style normalization here.
            v = self.matconvnetNormalize(v, 1e-12)
            v = tf.transpose(v, perm=[0, 2, 1])
            v = self.matconvnetNormalize(tf.layers.flatten(v), 1e-12)

        return v

    def matconvnetNormalize(self,inputs, epsilon):
        return inputs / tf.sqrt(tf.reduce_sum(inputs ** 2, axis=-1, keep_dims=True)
                                + epsilon)

    def compute_output_shape(self, input_shape):
        return tuple([None, self.outdim])
