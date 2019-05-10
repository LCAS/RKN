import numpy as np

"""Utility for pendulum experiments, allows for representation without discontinuities due to periodicity """

def to_angular_representation(angles):
    """maps theta -> [sin(theta), cos(theta)]
    :param angles, batch of angular values
    :returns batch of angles represented as [sin(theta), cos(theta)] """
    labels_sin = np.sin(angles)
    labels_cos = np.cos(angles)
    return np.concatenate([labels_sin, labels_cos], axis=-1)

def from_angular_representation(sin, cos):
    """ maps [sin(theta), cos(theta)] -> theta
    :param sin: sine of theta (batch)
    :param cos: cosine of theta (batch)
    :returns: theta (batch)"""
    sin = np.clip(sin, a_min=-1, a_max=1)
    cos = np.clip(cos, a_min=-1, a_max=1)
    angle = np.arccos(cos)
    angle[np.arcsin(sin) < 0] *= -1
    return angle