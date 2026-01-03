import numpy as np
from coffea.analysis_tools import Weights


def add_particle_shower_weight(
    events, weights_container: Weights, year: str, variation: str = "nominal"
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
    # add ps weights ISR
    weights_container.add(
        "ps_isr",
        weight=np.ones(len(events)),
        weightUp=events.PSWeight[:,0],
        weightDown=events.PSWeight[:,2],
    )

    # add ps weights FSR
    weights_container.add(
        "ps_fsr",
        weight=np.ones(len(events)),
        weightUp=events.PSWeight[:,1],
        weightDown=events.PSWeight[:,3],
    )    
