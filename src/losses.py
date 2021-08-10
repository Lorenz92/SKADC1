import tensorflow as tf
from keras import backend as K

import sys

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4

def rpn_loss_regr(num_anchors):
    """Loss function for rpn regression
    Args:
        num_anchors: number of anchors (9 in here)
    Returns:
        Smooth L1 loss function 
                           0.5*x*x (if x_abs < 1)
                           x_abx - 0.5 (otherwise)
    """
    def rpn_loss_regr_fixed_num(y_true, y_pred):

        # x is the difference between true value and predicted value
        x = y_true[:, :, :, 4 * num_anchors:] - y_pred

        # absolute value of x
        x_abs = K.abs(x)

        # If x_abs <= 1.0, x_bool = 1
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

        return lambda_rpn_regr * K.sum(
            y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])

    return rpn_loss_regr_fixed_num

def rpn_loss_cls(num_anchors):
    """Loss function for rpn classification
    Args:
        num_anchors: number of anchors (9 in here)
        y_true[:, :, :, :9]: [0,1,0,0,0,0,0,1,0] means only the second and the eighth box is valid which contains pos or neg anchor => isValid
        y_true[:, :, :, 9:]: [0,1,0,0,0,0,0,0,0] means the second box is pos and eighth box is negative
    Returns:
        lambda * sum((binary_crossentropy(isValid*y_pred,y_true))) / N
    """
    def rpn_loss_cls_fixed_num(y_true, y_pred):
        # tf.print('pred = ', y_pred.shape, output_stream=sys.stderr, sep=',')
        # tf.print('true = ', y_true.shape, output_stream=sys.stderr, sep=',')

        return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])
            

    return rpn_loss_cls_fixed_num

def detector_loss_regr(num_classes):
    """Loss function for rpn regression
    Args:
        num_anchors: number of anchors (9 in here)
    Returns:
        Smooth L1 loss function 
                           0.5*x*x (if x_abs < 1)
                           x_abs - 0.5 (otherwise)
    """
    def detector_loss_regr_fixed_num(y_true, y_pred):
        # tf.print('y_true shape = ', y_true.shape, output_stream=sys.stderr, sep=',', summarize=-1)
        # tf.print('y_true = ', y_true[0,...], output_stream=sys.stderr, sep=',', summarize=-1)
        # tf.print('y_pred shape = ', y_pred.shape, output_stream=sys.stderr, sep=',', summarize=-1)
        # tf.print('y_pred = ', y_pred[0,...], output_stream=sys.stderr, sep=',', summarize=-1)

        x = y_true[:, :, 4*num_classes:] - y_pred # Here <y_true[:, :, 4*num_classes:]> is the y_class_regr_coords of Y2 in calc_iou, so it represents the roi coordinates
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        z = lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes]) # Here <y_true[:, :, :4*num_classes]> is the y_class_regr_label of Y2 in calc_iou, so it represents a mask for the roi to be considered
        # tf.print('z = ', z, output_stream=sys.stderr, sep=',')
        return z
    return detector_loss_regr_fixed_num

def detector_loss_cls(y_true, y_pred):
    return lambda_cls_class * K.mean(K.categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))