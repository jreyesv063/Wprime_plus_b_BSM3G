import json
import numpy as np
import awkward as ak
import importlib.resources
from pathlib import Path
from coffea.nanoevents.methods.base import NanoEventsArray


def select_good_mt(
    met: ak.Array,
    lepton: ak.Array,
    mt_cut: float,
    invert_mt_cut: bool,
) -> ak.highlevel.Array:

    # Calculate lepton_met_mass
    lepton_met_mass = np.sqrt(
        2.0
        * lepton.pt
        * met.pt
        * (
            ak.ones_like(met.pt)
            - np.cos(lepton.delta_phi(met))
        )
    )

    # Determine the mask of events that meet the mt_min and mt_max conditions
    good_mt = (
        (lepton_met_mass >= mt_cut)
    )

    # Check if at least one of the values in each event has the condition
    good_mt_any = ak.any(good_mt, axis=-1)

    # Invert the mask if necessary: less than the mt_cut
    if invert_mt_cut:

        return ~good_mt_any
    
    else:
        
        return good_mt_any
