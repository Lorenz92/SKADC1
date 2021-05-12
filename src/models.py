import keras
from keras.layers import Input
from src.layers import Expander, vgg16, RpnNet


class VGG16(keras.Model):
    """
    VGG16 model wrapper
    """

    def __init__(self, anchor_num=30, **kwargs):
        # super(VGG16, self).__init__()
        self.anchor_num = anchor_num
        self.build(**kwargs)
    
    def call(self, inputs, training=False):

        # Expand image channels from 1 to 3
        x = Expander()(inputs)
        
        # Feature extraction network
        x = vgg16(x)

        # Create Region Proposal Net
        x = RpnNet(self.anchor_num)(x)

        if training:
            print('TRAINING') #training loop
        else:
            print('TEST')

        return x

    def build(self, shape, **kwargs):
        """
        Replicates Model(inputs=[inputs], outputs=[outputs]) of functional model.
        """
        # Replace with shape=[None, None, None, 1] if input_shape is unknown.
        inputs  = Input(shape=shape)
        outputs = self.call(inputs)
        super(VGG16, self).__init__(name="VGG16", inputs=inputs, outputs=outputs, **kwargs) 
