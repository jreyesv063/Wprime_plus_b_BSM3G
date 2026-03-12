import json
import numpy as np
import awkward as ak
from typing import Dict


def select_good_muons(
    events: ak.Array,
    muon_pt_threshold: float,
    muon_eta_threshold: float,
    muon_id_wp: str,
    muon_iso_wp: str,
    syst_var: bool = True    
) -> Dict[str, ak.Array]:
    """
    Build muon selection masks for nominal and systematic variations.

    Parameters
    ----------
    events : ak.Array
        NanoEvents object containing Muon collection.

    Returns
    -------
    Dict[str, ak.Array]
        Dictionary of boolean masks (per muon) for each variation.
    """

    # ------------------------------------------------------------------
    # Load working points from JSON
    # ------------------------------------------------------------------
    with open("wprime_plus_b/json_files/muon.json", "r") as f:
        muon_info = json.load(f)

    # ============================================================
    # Eta
    # ============================================================
    muon_eta_mask = (np.abs(events.Muon.eta) < muon_eta_threshold)

    # ============================================================
    # ID 
    # ============================================================
    muon_id_mask = getattr(events.Muon, muon_info['Id'][muon_id_wp])

    # ============================================================
    # Isolation
    # ============================================================
    muon_iso_mask = getattr(events.Muon, muon_info['Iso']['Flag']) < muon_info['Iso'][muon_iso_wp]

    # ============================================================
    # Pt
    # ============================================================
    muon_pt_mask = events.Muon.pt >= muon_pt_threshold
    
    # ============================================================
    # Final electron selection mask
    # ============================================================  
    muon_mask_ref = muon_eta_mask & muon_id_mask & muon_iso_mask

    muon_mask = muon_pt_mask & muon_mask_ref

    # ============================================================
    # Systematic variations: Rochester
    # ============================================================  
    # Check if the necessary attributes for Rochester variations are present in the events and Muon collection 
    if (
        syst_var 
        and hasattr(events, "genWeight")
        and hasattr(events.Muon, "pt_up")
        and hasattr(events.Muon, "pt_down")
    ):
        muon_up_mask = (events.Muon.pt_up >= muon_pt_threshold) & muon_mask_ref
        muon_down_mask = (events.Muon.pt_down >= muon_pt_threshold) & muon_mask_ref

        return {
            "nominal": muon_mask,
            "Rochester": {
                "up": muon_up_mask, "down": muon_down_mask
            }
        }

    else:
        return  {
            "nominal": muon_mask
        }

        