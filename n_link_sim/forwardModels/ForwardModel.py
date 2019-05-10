import _DoubleLinkForwardModel as DoubleLink
import _QuadLinkForwardModel as QuadLink
import numpy as np

# We create the array to return the values in here in python for two reasons:
# - We can only return arrays with swig whose size is known before the function call
# - By creating the array in python the interpreter keeps complete control over the object and the data hold by it

def simulate_double_link(states, actions, lengths, masses, inertias, g, friction, dt, dst,
                         use_pd=0, pdSetPoints=np.zeros((4)), pdGain=np.zeros((4))):
    if len(np.shape(states)) == 1:
        states = np.expand_dims(states, 0)
        actions = np.expand_dims(actions, 0)
    num_samples = len(states)
    if dt-1e-3 < 1.0 < dt+1e-3:
        result = np.zeros((num_samples, 2))
    else:
        result = np.zeros((num_samples, 6))
    DoubleLink.simulate(states, actions, dt, masses, lengths, inertias, g,
                        friction, dst, use_pd, pdSetPoints, pdGain, result)
    return result


def simulate_quad_link(states, actions, lengths, masses, inertias, g, friction, dt, dst):
    if len(np.shape(states)) == 1:
        states = np.expand_dims(states, 0)
        actions = np.expand_dims(actions, 0)
    num_samples = len(states)
    if dt-1e-3 < 1.0 < dt+1e-3:
        result = np.zeros((num_samples,  4))
    else:
        result = np.zeros((num_samples, 12))
    QuadLink.simulate(states, actions, dt, masses, lengths, inertias, g, friction, dst, result)
    return result

