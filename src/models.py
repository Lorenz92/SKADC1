import keras
from keras.layers import Input
from src.layers import Expander, vgg16, RpnNet, RoiPoolingConv, Classifier


class RpnVgg16(keras.Model):
    """
    VGG16 model wrapper
    """

    def __init__(self, anchor_num=30, **kwargs):
        # super(RpnVgg16, self).__init__()
        self.anchor_num = anchor_num
        self.build(**kwargs)
    
    def call(self, inputs, training=False):

        # Expand image channels from 1 to 3
        x = Expander()(inputs)
        
        # Feature extraction network
        x = vgg16(x)

        # Create Region Proposal Net
        rpn_cls, rpn_reg = RpnNet(self.anchor_num)(x)

        # if training:
        #     print('TRAINING') #training loop
        # else:
        #     print('TEST')

        return [rpn_cls, rpn_reg]

    def build(self, shape, **kwargs):
        """
        Replicates Model(inputs=[inputs], outputs=[outputs]) of functional model.
        """
        # Replace with shape=[None, None, None, 1] if input_shape is unknown.
        inputs  = Input(shape=shape)
        outputs = self.call(inputs)
        super(RpnVgg16, self).__init__(name="RpnVgg16", inputs=inputs, outputs=outputs, **kwargs)


def rpnvgg16(input_shape, anchor_num):

    # Add custom input-layer to change from1 to 3 channels
    input = Input(shape=input_shape)
    x = Expander()(input)

    # # Load pretrained VGG16 and remove last MaxPool layer
    x = vgg16(x)

    # Create Region Proposal Net
    rpn_cls, rpn_reg = RpnNet(anchor_num)(x)
    
    return keras.Model(input, [rpn_cls, rpn_reg], name='RegionProposal')



class ClsVgg16(keras.Model):
    """
    VGG16 model wrapper
    """

    def __init__(self, num_classes, **kwargs):
        # super(ClsVgg16, self).__init__()
        self.pooling_regions = 7
        self.num_rois = 4
        self.num_classes = num_classes
        self.build(**kwargs)
    
    def call(self, inputs, training=False):
        
        # Expand image channels from 1 to 3        
        x = Expander()(inputs[0])

        # Feature extraction network
        x = vgg16(x)
        
        # Final classification and regression step
        x = RoiPoolingConv(self.pooling_regions, self.num_rois)([x, inputs[1]])
        x = Classifier(self.num_classes)(x)

        # if training:
        #     print('TRAINING') #training loop
        # else:
        #     print('TEST')

        return x

    def build(self, shape_1, shape_2=(None, 4), **kwargs):
        """
        Replicates Model(inputs=[inputs], outputs=[outputs]) of functional model.
        """
        # Replace with shape=[None, None, None, 1] if input_shape is unknown.
        inputs  = [Input(shape=shape_1), Input(shape=shape_2)]
        outputs = self.call(inputs)
        super(ClsVgg16, self).__init__(name="ClsVgg16", inputs=inputs, outputs=outputs, **kwargs)


class TotalVgg16(keras.Model):
    """
    This model is the combination of RpnVgg16 and ClsVgg16, needed to save and load weights by name
    """

    def __init__(self, num_classes, anchor_num=30, **kwargs):
        # super(ClsVgg16, self).__init__()
        self.anchor_num = anchor_num
        self.pooling_regions = 7
        self.num_rois = 4
        self.num_classes = num_classes
        self.build(**kwargs)
    
    def call(self, inputs, training=False):
        
        # Expand image channels from 1 to 3        
        x = Expander()(inputs[0])

        # Feature extraction network
        x = vgg16(x)

        # Region Proposal Net
        rpn_cls, rpn_reg = RpnNet(self.anchor_num)(x)
        
        # Final classification and regression step
        x = RoiPoolingConv(self.pooling_regions, self.num_rois)([x, inputs[1]])
        x = Classifier(self.num_classes)(x)

        if training:
            print('TRAINING') #training loop
        else:
            print('TEST')

        return [rpn_cls, rpn_reg, x]

    def build(self, shape_1, shape_2=(None, 4), **kwargs):
        """
        Replicates Model(inputs=[inputs], outputs=[outputs]) of functional model.
        """
        # Replace with shape=[None, None, None, 1] if input_shape is unknown.
        inputs  = [Input(shape=shape_1), Input(shape=shape_2)]
        outputs = self.call(inputs)
        super(TotalVgg16, self).__init__(name="ClsVgg16", inputs=inputs, outputs=outputs, **kwargs)

