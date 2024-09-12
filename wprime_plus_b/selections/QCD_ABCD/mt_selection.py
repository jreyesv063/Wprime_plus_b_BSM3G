import json
import numpy as np
import awkward as ak
import importlib.resources
from pathlib import Path
from coffea.nanoevents.methods.base import NanoEventsArray


def select_good_mt(
    events: ak.Array,
    lepton: ak.Array,
    mt_min: float,
    mt_max: float,
    invert_mt_cut: bool,
) -> ak.highlevel.Array:

    # Calculate lepton_met_mass
    lepton_met_mass = np.sqrt(
        2.0
        * lepton.pt
        * events.MET.pt
        * (
            ak.ones_like(events.MET.pt)
            - np.cos(lepton.delta_phi(events.MET))
        )
    )

    # Determine the mask of events that meet the mt_min and mt_max conditions
    good_mt = (
        (lepton_met_mass >= mt_min)
        & (lepton_met_mass <= mt_max)
    )

    # Check if at least one of the values in each event has the condition
    good_mt_any = ak.any(good_mt, axis=-1)

    # Invert the mask if necessary
    if invert_mt_cut:

        return ~good_mt_any
    
    else:
        
        return good_mt_any
