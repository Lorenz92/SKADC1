import os
from src.layers import Expander, RpnNet, RoiPoolingConv, Detector, vgg16, resnet50, baseline_8, baseline_16, baseline_44
import src.config as config
import src.utils as utils
import keras
from keras.layers import Input
from keras.optimizers import Adam

def get_train_model(input_shape_1, input_shape_2, anchor_num, pooling_regions, num_rois, num_classes, backbone='vgg16', use_expander=True):

    
    if use_expander:    
        # Add custom input-layer to expand from 1 to 3 channels
        input_image  = Input(shape=(input_shape_1, input_shape_1, 1), name='input_1')
        x = Expander()(input_image)
    else:
        # In order to use this input layer we replicate the same input image 3 times because we have 1-channel images
        input_image  = Input(shape=(input_shape_1, input_shape_1, 3), name='input_1')
        x = input_image
        
    if backbone == 'vgg16':
        # Load pretrained VGG16 without last MaxPool layer
        x = vgg16(x)
    elif backbone == 'resnet50':
        x = resnet50(x, train_stage2 = False, train_stage3 = True, train_stage4 = True)
    elif backbone == 'baseline_8':
        x = baseline_8(x)
    elif backbone == 'baseline_16':
        x = baseline_16(x)
    elif backbone == 'baseline_44':
        x = baseline_44(x)

    # Create Region Proposal Net
    rpn_cls, rpn_reg, _ = RpnNet(anchor_num)(x)

    input_roi =  Input(shape=input_shape_2, name='input_2') # Here (None, 4)
    
    # Final classification and regression step
    x = RoiPoolingConv(pooling_regions, num_rois)([x, input_roi])
    out_class, out_regr = Detector(num_classes)(x)

    rpn_model = keras.Model(input_image, [rpn_cls, rpn_reg], name='RegionProposal')
    detector_model = keras.Model([input_image, input_roi], [out_class, out_regr], name='DetectorClassifier')
    total_model = keras.Model([input_image, input_roi], [rpn_cls, rpn_reg, out_class, out_regr], name='End2end_model')

    return rpn_model, detector_model, total_model

def get_eval_model(input_shape_1, input_shape_2, input_shape_fmap, anchor_num, pooling_regions, num_rois, num_classes, backbone='vgg16', use_expander=False):
    
    if use_expander:    
        # Add custom input-layer to expand from 1 to 3 channels
        input_image  = Input(shape=(input_shape_1, input_shape_1, 1), name='input_1')

        x = Expander()(input_image)
    else:
        # In order to use this input layer we replicate the same input image 3 times because we have 1-channel images
        input_image  = Input(shape=(input_shape_1, input_shape_1, 3), name='input_1')

        x = input_image

    if backbone == 'vgg16':
        x = vgg16(x)
        input_shape_fmap = (37, 37, 512)
    elif backbone == 'resnet50':
        x = resnet50(x)
        input_shape_fmap = (38, 38, 1024)
    elif backbone == 'baseline_8':
        x = baseline_8(x)        
        input_shape_fmap = (50, 50, 64)
    elif backbone == 'baseline_16':
        x = baseline_16(x)
        # input_shape_fmap = (37, 37, 128)
        input_shape_fmap = (25, 25, 128)
    elif backbone == 'baseline_44':
        x = baseline_44(x)
        # input_shape_fmap = (18, 18, 256)
        input_shape_fmap = (12, 12, 256)

    # Create Region Proposal Net
    rpn_cls, rpn_reg, shared_backbone = RpnNet(anchor_num)(x)

    input_roi = Input(shape=input_shape_2, name='input_2')
    input_fmap = Input(shape=input_shape_fmap, name='Input_fmap')
    
    # Final classification and regression step
    x = RoiPoolingConv(pooling_regions, num_rois)([input_fmap, input_roi])
    out_class, out_regr = Detector(num_classes)(x)

    rpn_model = keras.Model(input_image, [rpn_cls, rpn_reg, shared_backbone], name='RegionProposal')
    detector_model = keras.Model([input_fmap, input_roi], [out_class, out_regr], name='DetectorClassifier')
    total_model = keras.Model([input_image, input_roi, input_fmap], [rpn_cls, rpn_reg, out_class, out_regr], name='End2end_model')

    return rpn_model, detector_model, total_model
    

def load_weigths(rpn_model, detector_model, backbone, resume_train=True, checkpoint=None):

    if resume_train:
        if checkpoint is None:
            raise ValueError('Invalid "resume_train" and "checkpoint" combination...')
        weights = f'{config.MODEL_WEIGHTS}/{backbone}/{checkpoint}'

    else:
        if backbone == 'baseline_8':
            weights = f'{config.MODEL_WEIGHTS}/{backbone}/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
            # Download VGG16 weights
            # 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
            if not os.path.exists(weights):
                utils.download_data('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 
                'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', config.MODEL_WEIGHTS + f'/{backbone}')
        
        elif backbone == 'baseline_16':
            weights = f'{config.MODEL_WEIGHTS}/{backbone}/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
            # Download VGG16 weights
            # 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
            if not os.path.exists(weights):
                utils.download_data('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 
                'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', config.MODEL_WEIGHTS + f'/{backbone}')

        elif backbone == 'baseline_44':
            weights = f'{config.MODEL_WEIGHTS}/{backbone}/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
            # Download VGG16 weights
            # 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
            if not os.path.exists(weights):
                utils.download_data('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 
                'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', config.MODEL_WEIGHTS + f'/{backbone}')

        
        elif backbone == 'vgg16':
            weights = f'{config.MODEL_WEIGHTS}/{backbone}/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
            # Download VGG16 weights
            # 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
            if not os.path.exists(weights):
                utils.download_data('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 
                'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', config.MODEL_WEIGHTS + f'/{backbone}')
        
        elif backbone == 'resnet50':
            weights = f'{config.MODEL_WEIGHTS}/{backbone}/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
            # Download ResNet50 weights
            # 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
            if not os.path.exists(weights):
                utils.download_data('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', 
                'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', config.MODEL_WEIGHTS + f'/{backbone}')
            
        else:
            raise ValueError('Please choose a valid model')

    rpn_model.load_weights(weights, by_name=True)
    detector_model.load_weights(weights, by_name=True)
    return

def compile_models(rpn_model, detector_model, total_model, rpn_losses, detector_losses, class_list, rpn_lr=1e-2, rpn_clipnorm=0.001, detector_lr=1e-2, detector_clipnorm=0.001):

    ########## Build models

    # rpn_optimizer = Adam(lr=rpn_lr, clipnorm=rpn_clipnorm)
    # detector_optimizer = Adam(lr=detector_lr, clipnorm=detector_clipnorm)
    rpn_optimizer = Adam(lr=5e-2)
    detector_optimizer = Adam(lr=5e-2)

    rpn_loss_cls = rpn_losses[0]
    rpn_loss_regr = rpn_losses[1]
    detector_loss_cls = detector_losses[0]
    detector_loss_regr = detector_losses[1]

    rpn_model.compile(optimizer=rpn_optimizer, loss=[rpn_loss_cls(config.anchor_num), rpn_loss_regr(config.anchor_num)])
    detector_model.compile(optimizer=detector_optimizer, loss=[detector_loss_cls, detector_loss_regr(len(class_list))], metrics=['accuracy'])
    total_model.compile(optimizer='sgd', loss='mae')

    return