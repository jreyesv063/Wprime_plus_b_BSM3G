import numpy as np
import awkward as ak


def delta_r(first: ak.Array, second: ak.Array, threshold: float):
    """
    Calculates delta R and returns a mask for objects separated by more than threshold.
    delta_phi: https://github.com/scikit-hep/coffea/blob/1f69f3a373740b4f916139545cff7e2d87d08116/coffea/nanoevents/methods/vector.py#L67-L68
    """
    delta_eta = second.eta - first.eta
    delta_phi = (second.phi - first.phi + np.pi) % (2*np.pi) - np.pi
    delta_R = np.sqrt(delta_eta**2 + delta_phi**2)
    
    return delta_R > threshold