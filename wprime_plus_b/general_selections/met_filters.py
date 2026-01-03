import json
import numpy as np
import awkward as ak
import importlib.resources

def get_met_filters_mask(events: ak.Array, year: str, is_mc: bool) -> np.ndarray:
    """
    Build a boolean mask for events that pass the recommended MET filters.

    Parameters
    ----------
    events : ak.Array
        NanoEvents array containing 'Flag' fields (events.Flag).
    year : str
        Data-taking year, e.g., "2017", "2018".
    is_mc : bool
        True if processing MC. Determines which MET filter list to use.

    Returns
    -------
    metfilters : np.ndarray
        Boolean mask per event. True if the event passes all MET filters.
    """
    # Load MET filters from JSON
    with open("wprime_plus_b/json_files/metfilters.json", "r") as f:
        metfilters_json = json.load(f)[year]

    # Initialize mask: all events True
    metfilters = np.ones(len(events), dtype=bool)

    # Choose data or MC filters
    metfilterkey = "mc" if is_mc else "data"

    for mf in metfilters_json[metfilterkey]:
        if mf in events.Flag.fields:
            metfilters = metfilters & events.Flag[mf]

    return metfilters