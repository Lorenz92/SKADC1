# First net version
import sys

import keras.backend as K
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Layer, Flatten, TimeDistributed, Dense, Dropout, ZeroPadding2D, Convolution2D, BatchNormalization, Activation, Add
from tensorflow.keras import initializers

"""
rf = rf_l-1 + (s * k-1)
"""

kernel_init = initializers.RandomNormal(mean=.0, stddev=0.01)
bias_init = initializers.Zeros()

class Expander(Layer):
    def __init__(self, channels=3, kernel_size=(1, 1), padding='same', activation='relu', name='Custom_input_layer'):
        super(Expander, self).__init__(name=name)
        self.conv2d = Conv2D(filters=channels, kernel_size=kernel_size, padding=padding, activation=activation, name=name)

    def call(self, inputs):
        x = self.conv2d(inputs)
        return x

def baseline_8(input_image):

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(input_image) # RF = 3
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x) # RF = 5
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv3', trainable=True)(x) # RF = 7
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x) # RF = 8

    return x

def baseline_16(input_image):

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(input_image) # RF = 3
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x) # RF = 5
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x) # RF = 6

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=False)(x) # RF = 10
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=False)(x) # RF = 14
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x) # RF = 16

    # x = Conv2D(256, (1, 1), activation='relu', padding='same', name='cust_blockx_conv1', trainable=True)(x) # RF = 16
    # x = Conv2D(512, (1, 1), activation='relu', padding='same', name='cust_blockx_conv2', trainable=True)(x) # RF = 16

    return x

def baseline_44(input_image):

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(input_image) # RF = 3
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x) # RF = 5
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x) # RF = 6

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=False)(x) # RF = 10
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=False)(x) # RF = 14
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x) # RF = 16

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x) # RF = 24
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x) # RF = 32
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x) # RF = 40
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x) # RF = 44 # 36
    
    return x


def vgg16(input_image):

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(input_image) # RF = 3
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x) # RF = 5
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x) # RF = 6

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=False)(x) # RF = 10
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=False)(x) # RF = 14
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x) # RF = 16

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x) # RF = 24
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x) # RF = 32
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x) # RF = 40
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x) # RF = 44

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x) # RF = 60
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x) # RF = 76
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x) # RF = 92
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x) # RF = 100

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x) # RF = 132
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x) # RF = 164
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x) # RF = 196
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x) # RF = 212

    return x

def resnet50(input_image, train_stage2 = False, train_stage3 = False, train_stage4 = False):

    # Stage 1
    x = ZeroPadding2D((3, 3))(input_image)
    x = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1', trainable = False)(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable = train_stage2)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', trainable = train_stage2)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', trainable = train_stage2)

    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', trainable = train_stage3)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', trainable = train_stage3)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', trainable = train_stage3)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', trainable = train_stage3)

    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', trainable = train_stage4)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', trainable = train_stage4)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', trainable = train_stage4)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', trainable = train_stage4)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', trainable = train_stage4)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', trainable = train_stage4)

    return x

def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True):

    nb_filter1, nb_filter2, nb_filter3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), name=conv_name_base + '2a', trainable=trainable)(input_tensor)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable)(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):

    nb_filter1, nb_filter2, nb_filter3 = filters
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', trainable=trainable)(input_tensor)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable)(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', trainable=trainable)(input_tensor)
    shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

class RpnNet(Layer):

    def __init__(self, anchor_num):
        self.prob_pred_out = anchor_num
        self.coord_pred_out = 4 * anchor_num 
        super(RpnNet, self).__init__(name='rpn')
        self.conv2d = Conv2D(filters=512, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init, activation='relu', name='18_RPN_Conv1')
        self.cls_pred = Conv2D(filters=self.prob_pred_out, kernel_size=(1, 1), activation='sigmoid', kernel_initializer='uniform', bias_initializer=bias_init, name='19_Anchor_Cls_Conv')
        self.reg_pred = Conv2D(filters=self.coord_pred_out, kernel_size=(1, 1), activation='linear', kernel_initializer='zero', bias_initializer=bias_init, name='19_Anchor_Reg_Conv')
    
    def call(self, backbone):
        x = self.conv2d(backbone)
        cls_pred = self.cls_pred(x) #output of layer 20
        reg_pred = self.reg_pred(x) #output of layer 19
        return [cls_pred, reg_pred, backbone]


class RoiPoolingConv(Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, rows, cols, channels)`
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    '''
    def __init__(self, pool_size, num_rois, **kwargs):

        # self.dim_ordering = K.common.image_dim_ordering()
        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(name='roi_pooling', **kwargs)

    def build(self, input_shape):
        self.channels = input_shape[0][3]   

    def compute_output_shape(self):
        return None, self.num_rois, self.pool_size, self.pool_size, self.channels

    def call(self, x):

        assert(len(x) == 2)

        # x[0] is image with shape (rows, cols, channels)
        img = x[0]

        # x[1] is roi with shape (num_rois,4) with ordering (x,y,w,h)
        rois = x[1]

        outputs = []

        # tf.print('img shape = ', img.shape, output_stream=sys.stderr, sep=',')
        # tf.print('img = ', img[0,...,0], output_stream=sys.stderr, sep=',')

        for roi_idx in range(self.num_rois):

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            # Resized roi of the image to pooling size (7x7)
            # max_dim = K.maximum(w, h)
            # # tf.print('roi shape = ', rois[0, roi_idx, :], output_stream=sys.stderr, sep=',')
            # # tf.print('img = ', img[0, ..., 0], output_stream=sys.stderr, sep=',')

            # if max_dim > self.pool_size:

            #     rs = tf.image.resize(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size), method='bilinear', antialias=False)
            # else:
            
            #     offset = (self.pool_size - max_dim)//2
            #     rs = tf.image.pad_to_bounding_box(img[:, y:y+h, x:x+w, :], offset, offset, self.pool_size, self.pool_size)
            # #     # tf.print('rs shape = ', rs.shape, output_stream=sys.stderr, sep=',')
            # #     # tf.print('rs = ', rs[0,...,0], output_stream=sys.stderr, sep=',')

            rs = tf.image.resize(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size), method='bilinear', antialias=False)
            
            outputs.append(rs)
                

        final_output = K.concatenate(outputs, axis=0)

        # Reshape to (1, num_rois, pool_size, pool_size, channels)
        # Might be (1, 4, 7, 7, # feature maps)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.channels))

        # permute_dimensions is similar to transpose
        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output
    
    
    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Detector(Layer):

    def __init__(self, num_classes, **kwargs):
        super(Detector, self).__init__(name='cls')

        self.num_classes = num_classes
        
        self.td_flatten = TimeDistributed(Flatten(name='flatten'))
        self.td_fc_1 = TimeDistributed(Dense(4096, activation='relu', name='fc1'))
        self.td_do_1 = TimeDistributed(Dropout(0.5))
        self.td_fc_2 = TimeDistributed(Dense(4096, activation='relu', name='fc2'))
        self.td_do_2 = TimeDistributed(Dropout(0.5))

        self.td_fc_cls = TimeDistributed(Dense(self.num_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(self.num_classes))
        self.td_fc_reg = TimeDistributed(Dense(4 * (self.num_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(self.num_classes))
        # self.td_fc_reg = TimeDistributed(Dense(4 * (self.num_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(self.num_classes))

    def call(self, inputs):

        x = self.td_flatten(inputs)
        x = self.td_fc_1(x)
        x = self.td_do_1(x)
        x = self.td_fc_2(x)
        x = self.td_do_2(x)

        # There are two output layer
        # out_class: softmax acivation function for classify the class name of the object
        # out_regr: linear activation function for bboxes coordinates regression
        out_class = self.td_fc_cls(x)
        out_regr = self.td_fc_reg(x)
        
        return [out_class, out_regr]

