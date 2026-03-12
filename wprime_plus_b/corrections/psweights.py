import numpy as np
import awkward as ak
from typing import Type
from coffea.analysis_tools import Weights


def add_particle_shower_weight(
    events: ak.Array, 
    weights_container: Type[Weights], 
    year: str
):
    """
    add particle shower weight

    Parameters:
    -----------
        events:
            Events array
        weights_container:
            Weight object from coffea.analysis_tools
        year:
            dataset year {'2016', '2017', '2018'}
        variation:
            if 'nominal' (default) add 'nominal', 'up' and 'down'
            variations to weights container. else, add only 'nominal' weights.
    """
    if not hasattr(events, "PSWeight") or ak.all(ak.num(events.PSWeight, axis=1) != 4):
        ones = np.ones(len(events))
        for v in ["ps", "pf"]:
            weights_container.add(f"{v}_isr_{year}", weight= ones, weightUp=ones, weightDown=ones)
    
    else:
        # add ps weights (Initial state radiation)
        weights_container.add(
            f"ps_isr_{year}",
            weight=np.ones(len(events)),
            weightUp=events.PSWeight[:,0],
            weightDown=events.PSWeight[:,2]
        )
    
        # add ps weights FSR (Final state radiation)
        weights_container.add(
            f"ps_fsr_{year}",
            weight=np.ones(len(events)),
            weightUp=events.PSWeight[:,1],
            weightDown=events.PSWeight[:,3]
        )    
