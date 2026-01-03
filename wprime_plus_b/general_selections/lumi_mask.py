import pickle
import numpy as np
import awkward as ak
import importlib.resources

def get_lumi_mask(events: ak.Array, year: str, is_mc: bool) -> np.ndarray:
    """
    Return the luminosity mask for data, or a trivial mask for MC.

    Parameters
    ----------
    events : ak.Array
        NanoEvents array containing 'run' and 'luminosityBlock'.
    year : str
        Data-taking year (used to select the correct lumi mask for data).
    is_mc : bool
        True if processing MC; MC always returns a full True mask.

    Returns
    -------
    lumi_mask : np.ndarray
        Boolean mask per event. True if the event passes the luminosity mask.
    """
    # Load the luminosity masks dictionary from pickle
    with open("wprime_plus_b/general_selections/lumi_Certificates/lumi_masks.pkl", "rb") as f:
        lumi_masks_dict = pickle.load(f)

    if not is_mc:
        # Apply the data lumi mask function for the given year
        lumi_mask_func = lumi_masks_dict[year]
        lumi_mask = lumi_mask_func(events.run, events.luminosityBlock)
    else:
        # For MC, accept all events
        lumi_mask = np.ones(len(events), dtype=bool)

    return lumi_mask