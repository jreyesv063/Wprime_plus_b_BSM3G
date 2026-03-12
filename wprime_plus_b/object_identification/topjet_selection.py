import json
import numpy as np
import awkward as ak
from typing import Dict
import importlib.resources



def select_good_topjets(
    events: ak.Array,
    year: str = "2017",
    topjet_pt_threshold: float = 300.0,
    topjet_eta_threshold: float = 2.4,
    TvsQCD: str = "Tight",
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
        topjet_info = json.load(f)


    # ============================================================
    # Eta
    # ============================================================
    topjet_eta_mask = (np.abs(events.FatJet.eta) <= topjet_eta_threshold)

    
    # ============================================================
    # jet ID
    # ============================================================
    topjet_jetId_mask = events.FatJet.jetId >= topjet_info[year]["jet_id"]    


    # ============================================================
    # Top tagger
    # ============================================================
    topjet_tagger_mask = (events.FatJet.particleNet_TvsQCD >= topjet_info[year]["TvsQCD"][TvsQCD]["value"])
    
    # ============================================================
    # pt
    # ============================================================
    topjet_pt_mask = (events.FatJet.pt >= topjet_pt_threshold)


    # ============================================================
    # Final bjet selection mask
    # ============================================================ 
    topjet_mask_ref = topjet_eta_mask & topjet_jetId_mask & topjet_tagger_mask 

    topjet_mask = topjet_pt_mask & topjet_mask_ref

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
        topjet_JES_up_mask = ((events.FatJet.JES_pt_up >= topjet_pt_threshold) & topjet_mask_ref)
        topjet_JES_down_mask = ((events.FatJet.JES_pt_down >= topjet_pt_threshold) & topjet_mask_ref)


        # JER
        topjet_JER_up_mask = ((events.FatJet.JER_pt_up >= topjet_pt_threshold) & topjet_mask_ref)
        topjet_JER_down_mask = ((events.FatJet.JER_pt_down >= topjet_pt_threshold) & topjet_mask_ref)


        return {
            "nominal": topjet_mask,
            "JES": {
                "up":  topjet_JES_up_mask, "down":  topjet_JES_down_mask
            },
            "JER": {
                "up":  topjet_JER_up_mask, "down":  topjet_JER_down_mask
            }
        }

    else:
        return  {
            "nominal": topjet_mask
        }










