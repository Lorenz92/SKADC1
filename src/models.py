import keras
from keras.layers import Input
from src.layers import Expander, vgg16, RpnNet, RoiPoolingConv, Classifier


# class RpnVgg16(keras.Model):
#     """
#     VGG16 model wrapper
#     """

#     def __init__(self, anchor_num=30, **kwargs):
#         # super(RpnVgg16, self).__init__()
#         self.anchor_num = anchor_num
#         self.build(**kwargs)
    
#     def call(self, inputs, training=False):

#         # Expand image channels from 1 to 3
#         x = Expander()(inputs)
        
#         # Feature extraction network
#         x = vgg16(x)

#         # Create Region Proposal Net
#         rpn_cls, rpn_reg = RpnNet(self.anchor_num)(x)

#         # if training:
#         #     print('TRAINING') #training loop
#         # else:
#         #     print('TEST')

#         return [rpn_cls, rpn_reg]

#     def build(self, shape, **kwargs):
#         """
#         Replicates Model(inputs=[inputs], outputs=[outputs]) of functional model.
#         """
#         # Replace with shape=[None, None, None, 1] if input_shape is unknown.
#         inputs  = Input(shape=shape)
#         outputs = self.call(inputs)
#         super(RpnVgg16, self).__init__(name="RpnVgg16", inputs=inputs, outputs=outputs, **kwargs)


# def rpnvgg16(input_shape, shared_net, anchor_num):

#     # Add custom input-layer to change from1 to 3 channels
#     input = Input(shape=input_shape)
#     x = Expander()(input)

#     # # Load pretrained VGG16 and remove last MaxPool layer
#     x = shared_net(x)

#     # Create Region Proposal Net
#     rpn_cls, rpn_reg = RpnNet(anchor_num)(x)
    
#     return keras.Model(input, [rpn_cls, rpn_reg], name='RegionProposal')

# def get_rpn_model(shared_net, anchor_num):

#     # Create Region Proposal Net
#     rpn_cls, rpn_reg = RpnNet(anchor_num)(shared_net)
    
#     return keras.Model(input, [rpn_cls, rpn_reg], name='RegionProposal')


# Per usare un unico modello andrebbero riscritte tutte le funzioni numpy in tf
def e2e(input_shape_1, input_shape_2, anchor_num, pooling_regions, num_rois, num_classes, weights, feature_extractor='vgg16'):
    
    # Add custom input-layer to change from1 to 3 channels
    input_image  = Input(shape=input_shape_1, name='input_1')

    x = Expander()(input_image)

    if feature_extractor == 'vgg16':
        # Load pretrained VGG16 and remove last MaxPool layer
        x = vgg16(x)

    # Create Region Proposal Net
    rpn_cls, rpn_reg = RpnNet(anchor_num)(x)

    input_roi =  Input(shape=input_shape_2, name='input_2')
    
    # Final classification and regression step
    x = RoiPoolingConv(pooling_regions, num_rois)([x, input_roi])
    out_class, out_regr = Classifier(num_classes)(x)

    rpn_model = keras.Model(input_image, [rpn_cls, rpn_reg], name='RegionProposal')
    cls_model = keras.Model([input_image, input_roi], [out_class, out_regr], name='DetectorClassifier')
    total_model = keras.Model([input_image, input_roi], [rpn_cls, rpn_reg, out_class, out_regr], name='End2end_model')

    rpn_model.load_weights(weights, by_name=True)
    cls_model.load_weights(weights, by_name=True)

    return rpn_model, cls_model, total_model



# class ClsVgg16(keras.Model):
#     """
#     VGG16 model wrapper
#     """

#     def __init__(self, num_classes, **kwargs):
#         # super(ClsVgg16, self).__init__()
#         self.pooling_regions = 7
#         self.num_rois = 4
#         self.num_classes = num_classes
#         self.build(**kwargs)
    
#     def call(self, inputs, training=False):
        
#         # Expand image channels from 1 to 3        
#         x = Expander()(inputs[0])

#         # Feature extraction network
#         x = vgg16(x)
        
#         # Final classification and regression step
#         x = RoiPoolingConv(self.pooling_regions, self.num_rois)([x, inputs[1]])
#         x = Classifier(self.num_classes)(x)

#         # if training:
#         #     print('TRAINING') #training loop
#         # else:
#         #     print('TEST')

#         return x

#     def build(self, shape_1, shape_2=(None, 4), **kwargs):
#         """
#         Replicates Model(inputs=[inputs], outputs=[outputs]) of functional model.
#         """
#         # Replace with shape=[None, None, None, 1] if input_shape is unknown.
#         inputs  = [Input(shape=shape_1), Input(shape=shape_2)]
#         outputs = self.call(inputs)
#         super(ClsVgg16, self).__init__(name="ClsVgg16", inputs=inputs, outputs=outputs, **kwargs)

# def clsvgg16(input_shape_1, input_shape_2, shared_net, pooling_regions, num_rois, num_classes):

#     input_image  = Input(shape=input_shape_1)
#     input_roi =  Input(shape=input_shape_2)

#     # Expand image channels from 1 to 3        
#     x = Expander()(input_image)

#     # Feature extraction network
#     x = shared_net(x)
    
#     # Final classification and regression step
#     x = RoiPoolingConv(pooling_regions, num_rois)([x, input_roi])
#     x = Classifier(num_classes)(x)

#     return keras.Model([input_image, input_roi], x, name='DetectorClassifier')


# class TotalVgg16(keras.Model):
#     """
#     This model is the combination of RpnVgg16 and ClsVgg16, needed to save and load weights by name
#     """

#     def __init__(self, num_classes, anchor_num=30, **kwargs):
#         # super(ClsVgg16, self).__init__()
#         self.anchor_num = anchor_num
#         self.pooling_regions = 7
#         self.num_rois = 4
#         self.num_classes = num_classes
#         self.build(**kwargs)
    
#     def call(self, inputs, training=False):
        
#         # Expand image channels from 1 to 3        
#         x = Expander()(inputs[0])

#         # Feature extraction network
#         x = vgg16(x)

#         # Region Proposal Net
#         rpn_cls, rpn_reg = RpnNet(self.anchor_num)(x)
        
#         # Final classification and regression step
#         x = RoiPoolingConv(self.pooling_regions, self.num_rois)([x, inputs[1]])
#         x = Classifier(self.num_classes)(x)

#         if training:
#             print('TRAINING') #training loop
#         else:
#             print('TEST')

#         return [rpn_cls, rpn_reg, x]

#     def build(self, shape_1, shape_2=(None, 4), **kwargs):
#         """
#         Replicates Model(inputs=[inputs], outputs=[outputs]) of functional model.
#         """
#         # Replace with shape=[None, None, None, 1] if input_shape is unknown.
#         inputs  = [Input(shape=shape_1), Input(shape=shape_2)]
#         outputs = self.call(inputs)
#         super(TotalVgg16, self).__init__(name="ClsVgg16", inputs=inputs, outputs=outputs, **kwargs)

