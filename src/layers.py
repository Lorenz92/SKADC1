# First net version
import keras.backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, Layer

# ANCHOR_NUM = 30
# PROB_PRED_OUT = 2 * ANCHOR_NUM
# COORD_PRED_OUT = 4 * ANCHOR_NUM 

def vgg16(input_image):

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(input_image)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=False)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    return x

# class Vgg16(Layer):
#     def __init__(self):
#         super(Vgg16, self).__init__()
        
#         self.conv2d_64_3x3 = Conv2D(64, (3, 3), activation='relu', padding='same', trainable=False)
#         self.conv2d_128_3x3 = Conv2D(128, (3, 3), activation='relu', padding='same', trainable=False)
#         self.conv2d_256_3x3 = Conv2D(256, (3, 3), activation='relu', padding='same', )
#         self.conv2d_512_3x3 = Conv2D(512, (3, 3), activation='relu', padding='same', )
#         self.maxpool_2x2 = MaxPooling2D((2, 2), strides=(2, 2))        

#     def call(self, inputs):

#         # Block 1
#         x = self.conv2d_64_3x3(inputs)
#         x = self.conv2d_64_3x3(x)
#         # x = self.maxpool_2x2(x)

#         # # Block 2
#         # x = self.conv2d_128_3x3(x)
#         # x = self.conv2d_128_3x3(x)
#         # x = self.maxpool_2x2(x)

#         # # Block 3
#         # x = self.conv2d_256_3x3(x)
#         # x = self.conv2d_256_3x3(x)
#         # x = self.conv2d_256_3x3(x)
#         # x = self.maxpool_2x2(x)

#         # # Block 4
#         # x = self.conv2d_512_3x3(x)
#         # x = self.conv2d_512_3x3(x)
#         # x = self.conv2d_512_3x3(x)
#         # x = self.maxpool_2x2(x)

#         # # Block 5
#         # x = self.conv2d_512_3x3(x)
#         # x = self.conv2d_512_3x3(x)
#         # x = self.conv2d_512_3x3(x)
#         # # x = self.maxpool_2x2(x)
#         return x


# def rpn_net(input_tensor=None):

#     # Create Region Proposal Net
#     x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='18_RPN_Conv1')(input_tensor)
#     cls_pred = Conv2D(filters=PROB_PRED_OUT, kernel_size=(1, 1), padding='same', activation='sigmoid', name='19_Anchor_Cls_Conv')(x) #output of layer 20
#     reg_pred = Conv2D(filters=COORD_PRED_OUT, kernel_size=(1, 1), padding='same', activation='linear', name='19_Anchor_Reg_Conv')(x) #output of layer 19

#     return [cls_pred, reg_pred]


class RpnNet(Layer):

    def __init__(self, anchor_num):
        self.pro_pred_ou = 2 * anchor_num
        self.coord_pred_out = 4 * anchor_num 
        super(RpnNet, self).__init__()
        self.conv2d = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='18_RPN_Conv1')
        self.cls_pred = Conv2D(filters=self.pro_pred_ou, kernel_size=(1, 1), padding='same', activation='sigmoid', name='19_Anchor_Cls_Conv')
        self.reg_pred = Conv2D(filters=self.coord_pred_out, kernel_size=(1, 1), padding='same', activation='linear', name='19_Anchor_Reg_Conv')

    def call(self, inputs):
        x = self.conv2d(inputs)
        cls_pred = self.cls_pred(x) #output of layer 20
        reg_pred = self.reg_pred(x) #output of layer 19
        return [cls_pred, reg_pred]



# def custom_input_layer(input_shape, channels=3, kernel_size=(1, 1), padding='same', activation='relu', name='Custom_input_layer'):

#     print(input_shape)

#     inp = Input(input_shape)
#     x = Conv2D(filters=channels, kernel_size=kernel_size, padding=padding, activation=activation, name=name)(inp)

#     return inp, x



class Expander(Layer):
    def __init__(self, channels=3, kernel_size=(1, 1), padding='same', activation='relu', name='Custom_input_layer'):
        super(Expander, self).__init__()
        self.conv2d = Conv2D(filters=channels, kernel_size=kernel_size, padding=padding, activation=activation, name=name) #TODO: parametrizzare tutto

    def call(self, inputs):
        x = self.conv2d(inputs)
        return x