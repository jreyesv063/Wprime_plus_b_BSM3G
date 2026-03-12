import json
import pickle
import numpy as np
import awkward as ak
from coffea.lumi_tools import LumiMask

def get_lumi_mask(events: ak.Array, year: str) -> np.ndarray:
    """
    Return the luminosity mask for data, or a trivial mask for MC.

    """
    # MC → accept all events
    if hasattr(events, "genWeight"):
        return np.ones(len(events), dtype=bool)

    # ===================================
    # Load certificate names
    # ===================================
    with open("wprime_plus_b/json_files/lumi_certificates.json", "r") as f:
        certificate_name = json.load(f)
    
    # Create LumiMask object
    lumi_map = LumiMask(f"wprime_plus_b/general_selections/certificates/{certificate_name[year]}")    

    lumi_mask = lumi_map(events.run, events.luminosityBlock)
    
    return lumi_mask