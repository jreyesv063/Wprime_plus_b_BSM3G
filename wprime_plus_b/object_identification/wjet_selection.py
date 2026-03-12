import json
import numpy as np
import awkward as ak
from typing import Dict
import importlib.resources as resources


def select_good_wjets(
    events: ak.Array,
    year: str = "2017",
    wjet_pt_threshold: float = 300.0,
    wjet_eta_threshold: float = 2.4,
    WvsQCD: str = "Tight",
    syst_var: bool = True
) -> Dict[str, ak.Array]:
    """
    Build masks selecting good fat jets for nominal and JES/JER variations.

    Parameters
    ----------
    fatjets : ak.Array
        FatJet collection.
    year : str
        Data-taking year (used to read working points).
    fatjet_pt_threshold : float
        Minimum pT requirement for fat jets.
    fatjet_eta_threshold : float
        Maximum |eta| requirement for fat jets.
    TvsQCD : str
        ParticleNet TvsQCD working point.
    is_mc : bool
        Whether the sample is MC or data.

    Returns
    -------
    Dict[str, ak.Array]
        Dictionary of boolean masks for each systematic variation.
    """
    # ------------------------------------------------------------------
    # Load working points from JSON
    # ------------------------------------------------------------------
    with open("wprime_plus_b/json_files/fatjet.json", "r") as f:
        wjet_info = json.load(f)


    # ============================================================
    # Eta
    # ============================================================
    wjet_eta_mask = (np.abs(events.FatJet.eta) <= wjet_eta_threshold)

    
    # ============================================================
    # jet ID
    # ============================================================
    wjet_jetId_mask = events.FatJet.jetId >= wjet_info[year]["jet_id"]    


    # ============================================================
    # Top tagger
    # ============================================================
    wjet_tagger_mask = (events.FatJet.particleNet_TvsQCD >= wjet_info[year]["WvsQCD"][WvsQCD]["value"])
    
    # ============================================================
    # pt
    # ============================================================
    wjet_pt_mask = (events.FatJet.pt >= wjet_pt_threshold)


    # ============================================================
    # Final bjet selection mask
    # ============================================================ 
    wjet_mask_ref = wjet_eta_mask & wjet_jetId_mask & wjet_tagger_mask 

    wjet_mask = wjet_pt_mask & wjet_mask_ref

    
    # ============================================================
    # Systematic variations: JES and JEC
    # ============================================================    
    if (
        syst_var
        and hasattr(events, "genWeight")
        and hasattr(events.FatJet, "JES_pt_up")
        and hasattr(events.FatJet, "JES_pt_down")
        and hasattr(events.FatJet, "JER_pt_up")
        and hasattr(events.FatJet, "JER_pt_down")
    ):  
        # JES
        wjet_JES_up_mask = ((events.FatJet.JES_pt_up >= wjet_pt_threshold) & wjet_mask_ref)
        wjet_JES_down_mask = ((events.FatJet.JES_pt_down >= wjet_pt_threshold) & wjet_mask_ref)


        # JER
        wjet_JER_up_mask = ((events.FatJet.JER_pt_up >= wjet_pt_threshold) & wjet_mask_ref)
        wjet_JER_down_mask = ((events.FatJet.JER_pt_down >= wjet_pt_threshold) & wjet_mask_ref)


        return {
            "nominal": wjet_mask,
            "JES": {
                "up":  wjet_JES_up_mask, "down": wjet_JES_down_mask
            },
            "JER": {
                "up":  wjet_JER_up_mask, "down":  wjet_JER_down_mask
            }
        }

    else:
        return  {
            "nominal": wjet_mask
        }
