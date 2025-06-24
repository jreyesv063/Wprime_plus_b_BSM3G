import json
import numpy as np
import awkward as ak
import importlib.resources
from pathlib import Path
from coffea.nanoevents.methods.base import NanoEventsArray


def select_good_delta_phi_jet_met(
    met: ak.Array,
    jets: ak.Array,
    delta_phi_cut: float,
    invert_delta_phi_cut: bool,
) -> ak.highlevel.Array:
    # --------------------------
    # TLorentz vectors: https://coffea-hep.readthedocs.io/en/v0.7.23/api/coffea.nanoevents.methods.vector.LorentzVector.html
    # Vectors: https://coffea-hep.readthedocs.io/en/v0.7.23/api/coffea.nanoevents.methods.vector.TwoVector.html#coffea.nanoevents.methods.vector.TwoVector.delta_phi
    # delta_phi is defined between [-pi, pi) ->  (a - b + numpy.pi) % (2 * numpy.pi) - numpy.pi
    # --------------------------     
    
    # delta_phi variable
    delta_phi_met_jet = jets.delta_phi(met)

    # Determine the mask of events that meet the mt_min and mt_max conditions
    good_delta_phi = (
        (ak.all(np.abs(delta_phi_met_jet) >= delta_phi_cut, axis = -1))
    )

    # Invert the mask if necessary: less than the mt_cut
    if invert_delta_phi_cut:

        return ~good_delta_phi
    
    else:
        
        return good_delta_phi
