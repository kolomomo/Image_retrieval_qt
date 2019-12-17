
from keras.layers import UpSampling2D
from keras.layers import MaxPool2D
from keras.layers import Add
from keras.layers import Multiply
from keras.layers import Lambda
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import concatenate
from keras.layers import add
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
# Backend
from keras import backend as K
from keras import regularizers
from keras import initializers
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense,Reshape,multiply

def Conv_bn_relu(num_filters,
                 kernel_size,
                 batchnorm=True,
                 strides=(1, 1),
                 padding='same'):

    def layer(input_tensor):
        x = Conv2D(num_filters, kernel_size,
                   padding=padding, kernel_initializer='he_normal',
                   strides=strides)(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    return layer

def conv2d_bn(x, nb_filter, num_row, num_col,
              padding='same', strides=(1, 1), use_bias=False):
    """
    Utility function to apply conv + BN.
    (Slightly modified from https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py)
    """
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    x = Convolution2D(nb_filter, (num_row, num_col),
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      kernel_regularizer=regularizers.l2(0.00004),
                      kernel_initializer=initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None))(x)
    x = BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(x)
    x = Activation('relu')(x)
    return x

def residual_block(input, input_channels=None, output_channels=None, kernel_size=(3, 3), stride=1):
    """
    full pre-activation residual block
    https://arxiv.org/pdf/1603.05027.pdf
    """
    if output_channels is None:
        output_channels = input.get_shape()[-1].value
    if input_channels is None:
        input_channels = output_channels // 4

    strides = (stride, stride)

    x = BatchNormalization()(input)
    x = Activation('relu')(x)
    x = Conv2D(input_channels, (1, 1))(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(input_channels, kernel_size, padding='same', strides=stride)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(output_channels, (1, 1), padding='same')(x)

    if input_channels != output_channels or stride != 1:
        input = Conv2D(output_channels, (1, 1), padding='same', strides=strides)(input)

    x = Add()([x, input])
    return x


def attention_block(input, input_channels=None, output_channels=None, encoder_depth=1):
    """
    attention block
    https://arxiv.org/abs/1704.06904
    """

    p = 1
    t = 2
    r = 1

    if input_channels is None:
        input_channels = input.get_shape()[-1].value
    if output_channels is None:
        output_channels = input_channels

    # First Residual Block
    for i in range(p):
        input = residual_block(input)

    # Trunc Branch
    output_trunk = input
    for i in range(t):
        output_trunk = residual_block(output_trunk)

    # Soft Mask Branch

    ## encoder
    ### first down sampling
    output_soft_mask = MaxPool2D(padding='same')(input)  # 32x32
    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)

    skip_connections = []
    for i in range(encoder_depth - 1):

        ## skip connections
        output_skip_connection = residual_block(output_soft_mask)
        skip_connections.append(output_skip_connection)
        # print ('skip shape:', output_skip_connection.get_shape())

        ## down sampling
        output_soft_mask = MaxPool2D(padding='same')(output_soft_mask)
        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)

            ## decoder
    skip_connections = list(reversed(skip_connections))
    for i in range(encoder_depth - 1):
        ## upsampling
        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)
        output_soft_mask = UpSampling2D()(output_soft_mask)
        ## skip connections
        output_soft_mask = Add()([output_soft_mask, skip_connections[i]])

    ### last upsampling
    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)
    output_soft_mask = UpSampling2D()(output_soft_mask)

    ## Output
    output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Activation('sigmoid')(output_soft_mask)

    # Attention: (1 + output_soft_mask) * output_trunk
    output = Lambda(lambda x: x + 1)(output_soft_mask)
    output = Multiply()([output, output_trunk])  #

    # Last Residual Block
    for i in range(p):
        output = residual_block(output)

    return output

def attention_block_a(input, input_channels=None, output_channels=None, encoder_depth=1):

    if input_channels is None:
        input_channels = input.get_shape()[-1].value
    if output_channels is None:
        output_channels = input_channels

    output_trunk = input
    output_trunk = block_inception_a(output_trunk)

    # Soft Mask Branch
    ## encoder
    ### first down sampling
    # output_soft_mask = MaxPool2D(padding='same')(input)  # 32x32
    output_soft_mask = input
    skip_connections = []
    for i in range(encoder_depth):
        ## skip connections
        output_skip_connection = conv2d_bn(output_soft_mask, 96, 3, 3)
        skip_connections.append(output_skip_connection)
        print ('skip shape:', output_skip_connection.get_shape())
        ## down sampling
        output_soft_mask = MaxPool2D(padding='same')(output_skip_connection)

    ## decoder
    skip_connections = list(reversed(skip_connections))
    for i in range(encoder_depth):
        ## upsampling
        output_soft_mask = conv2d_bn(output_soft_mask, 96, 3, 3)
        output_soft_mask = UpSampling2D()(output_soft_mask)
        ## skip connections
        print ('up skip shape:', output_soft_mask.get_shape())
        output_soft_mask = Add()([output_soft_mask, skip_connections[i]])
       ### last upsampling

    ## Output
    output_soft_mask = Conv2D(int(output_trunk.get_shape()[-1]), (1, 1))(output_soft_mask)
    output_soft_mask = Activation('sigmoid')(output_soft_mask)

    # Attention: (1 + output_soft_mask) * output_trunk
    output = Lambda(lambda x: x + 1)(output_soft_mask)
    output = Multiply()([output, output_trunk])  #

    # Last se Block
    output = seblock(output)
    # for i in range(p):
    #     output = residual_block(output)

    return output
def attention_block_b(input, input_channels=None, output_channels=None, encoder_depth=1):

    if input_channels is None:
        input_channels = input.get_shape()[-1].value
    if output_channels is None:
        output_channels = input_channels

    output_trunk = input
    output_trunk = block_inception_b(output_trunk)

    # Soft Mask Branch
    ## encoder
    ### first down sampling
    # output_soft_mask = MaxPool2D(padding='same')(input)  # 32x32
    output_soft_mask = input
    skip_connections = []
    for i in range(encoder_depth):
        ## skip connections
        output_skip_connection = conv2d_bn(output_soft_mask, 96, 3, 3)
        skip_connections.append(output_skip_connection)
        print ('skip shape:', output_skip_connection.get_shape())
        ## down sampling
        output_soft_mask = MaxPool2D(padding='same')(output_skip_connection)

    ## decoder
    skip_connections = list(reversed(skip_connections))
    for i in range(encoder_depth):
        ## upsampling
        output_soft_mask = conv2d_bn(output_soft_mask, 96, 3, 3)
        output_soft_mask = UpSampling2D()(output_soft_mask)
        ## skip connections
        print ('up skip shape:', output_soft_mask.get_shape())
        output_soft_mask = Add()([output_soft_mask, skip_connections[i]])
       ### last upsampling

    ## Output
    output_soft_mask = Conv2D(int(output_trunk.get_shape()[-1]), (1, 1))(output_soft_mask)
    output_soft_mask = Activation('sigmoid')(output_soft_mask)

    # Attention: (1 + output_soft_mask) * output_trunk
    output = Lambda(lambda x: x + 1)(output_soft_mask)
    output = Multiply()([output, output_trunk])  #

    # Last se Block
    output = seblock(output)
    # for i in range(p):
    #     output = residual_block(output)

    return output
def attention_block_c(input, input_channels=None, output_channels=None, encoder_depth=1):

    if input_channels is None:
        input_channels = input.get_shape()[-1].value
    if output_channels is None:
        output_channels = input_channels

    output_trunk = input
    output_trunk = block_inception_c(output_trunk)

    # Soft Mask Branch
    ## encoder
    ### first down sampling
    # output_soft_mask = MaxPool2D(padding='same')(input)  # 32x32
    output_soft_mask = input
    skip_connections = []
    for i in range(encoder_depth):
        ## skip connections
        output_skip_connection = conv2d_bn(output_soft_mask, 96, 3, 3)
        skip_connections.append(output_skip_connection)
        print ('skip shape:', output_skip_connection.get_shape())
        ## down sampling
        output_soft_mask = MaxPool2D(padding='same')(output_skip_connection)

    ## decoder
    skip_connections = list(reversed(skip_connections))
    for i in range(encoder_depth):
        ## upsampling
        output_soft_mask = conv2d_bn(output_soft_mask, 96, 3, 3)
        output_soft_mask = UpSampling2D()(output_soft_mask)
        ## skip connections
        print ('up skip shape:', output_soft_mask.get_shape())
        output_soft_mask = Add()([output_soft_mask, skip_connections[i]])
       ### last upsampling

    ## Output
    output_soft_mask = Conv2D(int(output_trunk.get_shape()[-1]), (1, 1))(output_soft_mask)
    output_soft_mask = Activation('sigmoid')(output_soft_mask)

    # Attention: (1 + output_soft_mask) * output_trunk
    output = Lambda(lambda x: x + 1)(output_soft_mask)
    output = Multiply()([output, output_trunk])  #

    # Last se Block
    output = seblock(output)
    # for i in range(p):
    #     output = residual_block(output)

    return output

def attention_block_X(input, input_channels=None, output_channels=None, encoder_depth=1, include_de=True, include_se=False):

    if input_channels is None:
        input_channels = input.get_shape()[-1].value
    if output_channels is None:
        output_channels = input_channels

    output_trunk = input
    output_trunk = block_inception_c(output_trunk)

    # Soft Mask Branch
    ## encoder
    ### first down sampling
    # output_soft_mask = MaxPool2D(padding='same')(input)  # 32x32
    if include_de:
        output_soft_mask = input
        skip_connections = []
        for i in range(encoder_depth):
            ## skip connections
            output_skip_connection = conv2d_bn(output_soft_mask, 96, 3, 3)
            skip_connections.append(output_skip_connection)
            print ('skip shape:', output_skip_connection.get_shape())
            ## down sampling
            output_soft_mask = MaxPool2D(padding='same')(output_skip_connection)

        ## decoder
        skip_connections = list(reversed(skip_connections))
        for i in range(encoder_depth):
            ## upsampling
            output_soft_mask = conv2d_bn(output_soft_mask, 96, 3, 3)
            output_soft_mask = UpSampling2D()(output_soft_mask)
            ## skip connections
            print ('up skip shape:', output_soft_mask.get_shape())
            output_soft_mask = Add()([output_soft_mask, skip_connections[i]])
           ### last upsampling

        ## Output
        output_soft_mask = Conv2D(int(output_trunk.get_shape()[-1]), (1, 1))(output_soft_mask)
        output_soft_mask = Activation('sigmoid')(output_soft_mask)

        # Attention: (1 + output_soft_mask) * output_trunk
        output = Lambda(lambda x: x + 1)(output_soft_mask)
        output = Multiply()([output, output_trunk])  #
    else:
        output = output_trunk
    if include_se:
        # Last se Block
        output = seblock(output)
        # for i in range(p):
        #     output = residual_block(output)
    return output

def block_inception_c(input):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 256, 1, 1)

    branch_1 = conv2d_bn(input, 384, 1, 1)
    branch_10 = conv2d_bn(branch_1, 256, 1, 3)
    branch_11 = conv2d_bn(branch_1, 256, 3, 1)
    branch_1 = concatenate([branch_10, branch_11], axis=channel_axis)


    branch_2 = conv2d_bn(input, 384, 1, 1)
    branch_2 = conv2d_bn(branch_2, 448, 3, 1)
    branch_2 = conv2d_bn(branch_2, 512, 1, 3)
    branch_20 = conv2d_bn(branch_2, 256, 1, 3)
    branch_21 = conv2d_bn(branch_2, 256, 3, 1)
    branch_2 = concatenate([branch_20, branch_21], axis=channel_axis)

    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    branch_3 = conv2d_bn(branch_3, 256, 1, 1)

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
    return x
def block_inception_b(input):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 384, 1, 1)

    branch_1 = conv2d_bn(input, 192, 1, 1)
    branch_1 = conv2d_bn(branch_1, 224, 1, 7)
    branch_1 = conv2d_bn(branch_1, 256, 7, 1)

    branch_2 = conv2d_bn(input, 192, 1, 1)
    branch_2 = conv2d_bn(branch_2, 192, 7, 1)
    branch_2 = conv2d_bn(branch_2, 224, 1, 7)
    branch_2 = conv2d_bn(branch_2, 224, 7, 1)
    branch_2 = conv2d_bn(branch_2, 256, 1, 7)

    branch_3 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input)
    branch_3 = conv2d_bn(branch_3, 128, 1, 1)

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
    return x
def block_inception_a(input):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 96, 1, 1)

    branch_1 = conv2d_bn(input, 64, 1, 1)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3)

    branch_2 = conv2d_bn(input, 64, 1, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3)

    branch_3 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input)
    branch_3 = conv2d_bn(branch_3, 96, 1, 1)

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
    return x

def slice_layer(x, slice_num, channel_input):
    output_list = []
    single_channel = channel_input//slice_num
    for i in range(slice_num):
        out = x[:, :, :, i*single_channel:(i+1)*single_channel]
        output_list.append(out)
    return output_list

def res2net_block(num_filters, slice_num):
    def layer(input_tensor):
        short_cut = input_tensor
        x = Conv_bn_relu(num_filters=num_filters, kernel_size=(1, 1))(input_tensor)
        slice_list = slice_layer(x, slice_num, x.shape[-1])
        side = Conv_bn_relu(num_filters=num_filters//slice_num, kernel_size=(3, 3))(slice_list[1])
        z = concatenate([slice_list[0], side])   # for one and second stage
        for i in range(2, len(slice_list)):
            y = Conv_bn_relu(num_filters=num_filters//slice_num, kernel_size=(3, 3))(add([side, slice_list[i]]))
            side = y
            z = concatenate([z, y])
        z = Conv_bn_relu(num_filters=num_filters, kernel_size=(1, 1))(z)
        out = concatenate([z, short_cut])
        return out
    return layer

def seblock(x):

    squeeze = GlobalAveragePooling2D()(x)
    u = int(x.shape[-1])
    excitation = Dense(units=u // 16)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=u)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1, 1, u))(excitation)

    scale = multiply([x, excitation])

    # x = GlobalAveragePooling2D()(scale)
    #
    # dp_1 = Dropout(0.6)(x)
    # fc2 = Dense(out_dims)(dp_1)
    # fc2 = Activation('sigmoid')(fc2) #此处注意，为sigmoid函数
    # model = Model(inputs=inputs_dim, outputs=fc2)
    return scale


#
# x = Input((256, 256, 256))
# print(x.shape)
# x_conv_nor = Conv_bn_relu(512, (3, 3))(x)
# print(x_conv_nor.shape)
# out = slice_layer(x_conv_nor, 8, 512)
# print(out)
# print(len(out))
# x = res2net_block(512, 8)(x_conv_nor)
# print(x.shape)


