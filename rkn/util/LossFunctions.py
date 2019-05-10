import tensorflow as tf
import numpy as np

NLL_NORM_CONSTANT = np.log(2*np.pi)

def mse(targets, predictions):
    """ Computes Mean Squared Error
    :param targets:
    :param predictions:
    :return:
    """
    return tf.reduce_mean(tf.square(predictions - targets))

def rmse(targets, predictions):
    """ Computes Root Mean Squared Error
    :param targets:
    :param predictions:
    :return:
    """
    rmse = tf.sqrt(tf.reduce_mean(tf.square(predictions-targets)))
    return rmse

def binary_crossentropy(targets, predictions, epsilon=1e-8, img_data=False, scale_targets=None):
    """ Computes Binary Cross Entropy
    :param targets:
    :param predictions:
    :param epsilon: offset for numerical stability (default = 1e-8)
    :param img_data: If true each sample is assumed to be an image (i.e. a 3D tensor) - else only a 1D Vector
    :param factor to scale the targets (if they are not inherently in the interval [0, 1]
    :return: Binary Crossentropy between targets and prediction
    """
    if scale_targets:
        targets = targets / scale_targets
    point_wise_error = - (targets * tf.log(predictions + epsilon) + (1 - targets) * tf.log(1 - predictions + epsilon))
    reduction = (-3, -2, -1) if img_data else (-1)
    sample_wise_error = tf.reduce_sum(point_wise_error, axis=reduction)
    return tf.reduce_mean(sample_wise_error)

def gaussian_nll(targets, predictions, variance, epsilon=1e-8, img_data=False, scale_targets=None):
    if scale_targets:
        targets = targets / scale_targets
    if img_data:
        variance = tf.expand_dims(tf.expand_dims(variance, -1), -1)
        sample_reduction = (-3, -2, -1)
    else:
        sample_reduction = -1
    variance += epsilon
    element_wise_nll = 0.5 * (NLL_NORM_CONSTANT + tf.log(variance) + ((targets - predictions)**2) / variance)
    sample_wise_error = tf.reduce_sum(element_wise_nll, axis=sample_reduction)
    return tf.reduce_mean(sample_wise_error)

def full_multivariate_gaussian_nll(targets, predictions, variance):
    dim = tf.shape(targets)[-1]
    constant_term = dim * NLL_NORM_CONSTANT
    regularization_term = tf.log(tf.matrix_determinant(variance))
    difference = predictions - targets
    m1 = tf.matrix_solve(variance, difference)
    m2 = tf.tensordot(difference, m1, -1)
    return 0.5 * (constant_term + regularization_term + m2)

