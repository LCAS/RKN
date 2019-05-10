import tensorflow as tf

"""Utility to concatenate and extract the mean and covariance vectors"""

def pack(mean, covar):
    """ Concatenates mean and covariance vector (works with batches)"""
    return tf.concat([mean, covar], -1)

def unpack(vector, dim):
    return vector[..., :dim], vector[..., dim:]


def pack_full_covar(mean, full_covar):
    dim = mean.get_shape()[-1].value
    shape = tf.shape(mean)
    rank = len(mean.get_shape())
    if rank == 2:
        covar_flat = tf.reshape(full_covar, shape=[shape[0], dim**2])
    elif rank == 3:
        covar_flat = tf.reshape(full_covar, shape=[shape[0], shape[1], dim**2])
    else:
        raise AssertionError("Pack Full Covar only for rank 2 and 3 supported")
    return tf.concat([mean, covar_flat], -1)

def unpack_full_covar(vector, dim):
    shape = tf.shape(vector)
    rank = len(vector.get_shape())
    if rank == 2:
        mean = vector[..., :dim]
        covar_flat = vector[:, dim:]
        covar = tf.reshape(covar_flat, shape=[shape[0], dim, dim])
    elif rank == 3:
        mean = vector[:, :, :dim]
        covar_flat = vector[:, :, dim:]
        covar = tf.reshape(covar_flat, shape=[shape[0], shape[1], dim, dim])
    else:
        raise AssertionError("Unpack Full Covar only for rank 2 and 3 supported")
    return mean, covar

def split_corr_covar(covar, dim):
    dim = int(dim / 2)
    assert covar.shape[-1] == 3 * dim
    return covar[..., 0 * dim : 1 * dim], \
           covar[..., 1 * dim : 2 * dim], \
           covar[..., 2 * dim : 3 * dim]

def pack_corr_covar(mean, covar):
    if isinstance(covar, list):
        return tf.concat([mean]  + covar, -1)
    else:
        return tf.concat([mean, covar], -1)

