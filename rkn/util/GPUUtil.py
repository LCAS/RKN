from tensorflow.python.client import device_lib
import numpy as np

"""Utility for working with GPUs"""

def print_devices():
    """Prints list of all available devices"""
    local_device_protos = device_lib.list_local_devices()
    print("Devices")
    for x in local_device_protos:
        print(x.name)

def get_available_gpus():
    """Returns names of all available devices"""
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_num_gpus():
    """Returns number of available devices """
    return len(get_available_gpus())

def _permute_channels(images, realign, permutation):
    """
    Permutes the channels of (batches of) images
    :param images: the images
    :param realign: whether to realign the images in memory
    :param permutation: the permutation of the channels
    :return:
    """
    b = np.transpose(images, axes=permutation)
    if realign:
        return np.ascontiguousarray(b)
    else:
        return b

def put_channels_first(images, realign=True):
    """Changes image representation from NHWC (intel/eigen default) to NCWH (cuda default)"""
    permutation = (0, 1, 4, 2, 3)
    return _permute_channels(images, realign, permutation)


def put_channels_last(images, realign=True):
    """Changes image representation from NCHW (cuda default) to NHWC (intel/eigen default)"""
    permutation = (0, 1, 3, 4, 2)
    return _permute_channels(images, realign, permutation)

def adapt_shape_for_gpu(shape):
    """Changes shape from NHWC to NCHW format
    :param shape: shape to change"""
    assert len(shape) >= 3, "Shape needs at least 3 elements to be adapted"
    return shape[:-3] + [shape[-1], shape[-3], shape[-2]]