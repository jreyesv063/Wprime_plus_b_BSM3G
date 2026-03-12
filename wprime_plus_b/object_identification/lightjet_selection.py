import json
import numpy as np
import awkward as ak
from typing import Dict
import importlib.resources
from typing import Optional


def select_good_lightjets(
    events: ak.Array,
    year: str = "2017",
    btag_working_point_fail: str = "Loose", 
    lightjet_pt_threshold: int = 20,
    lightjet_eta_threshold: float = 2.4,
    lightjet_id_wp: str = "TightLepVeto",
    lightjet_pileup_id: str = "Tight",
    syst_var: bool = True
) -> Dict[str, ak.Array]:
    
    # ------------------------------------------------------------------
    # Load working points from JSON
    # ------------------------------------------------------------------
    with open("wprime_plus_b/json_files/jet.json", "r") as f:
        lightjet_info = json.load(f)

    # --------------------------------------------------
    # Load b-tag thresholds
    # --------------------------------------------------
    with open("wprime_plus_b/json_files/bjet.json", "r") as f:
        btag_thresholds = json.load(f)["deepJet"][year]

        btag_fail = btag_thresholds[btag_working_point_fail]


    # ============================================================
    # Eta
    # ============================================================
    lightjet_eta_mask = np.abs(events.Jet.eta) < lightjet_eta_threshold   

    # ============================================================
    # jet ID
    # ============================================================
    if year in ["2016APV", "2016", "2017", "2018"]:
        lightjet_jetId_mask = (events.Jet.jetId  == lightjet_info["jetid"][year][lightjet_id_wp])


    # ============================================================
    # Btag flavor
    # ============================================================
    lightjet_btag_mask = (events.Jet.btagDeepFlavB < btag_fail)

    
    # ============================================================
    # pt
    # ============================================================
    lightjet_pt_mask = (events.Jet.pt >= lightjet_pt_threshold)

    
    # ============================================================
    # pileupjet ID (pt < 50)
    # ============================================================
    lightjet_pileupId_mask = (events.Jet.puId  == lightjet_info["pujetid"][year][lightjet_pileup_id])  
    
    
    # ============================================================
    # Final lightjet selection mask
    # ============================================================ 
    lightjet_mask_ref = lightjet_eta_mask & lightjet_jetId_mask & lightjet_btag_mask


    lightjet_mask = ak.where(
        (events.Jet.pt < 50),
        lightjet_mask_ref & lightjet_pileupId_mask & lightjet_pt_mask,
        lightjet_mask_ref & lightjet_pt_mask,
    )


    # ============================================================
    # Systematic variations: JES and JEC
    # ============================================================    
    if (
        syst_var
        and hasattr(events, "genWeight")
        and hasattr(events.Jet, "JES_pt_up")
        and hasattr(events.Jet, "JES_pt_down")
        and hasattr(events.Jet, "JER_pt_up")
        and hasattr(events.Jet, "JER_pt_down")
    ):  
        # JES
        lightjet_JES_up_mask = ak.where(
            (events.Jet.JES_pt_up < 50),
            lightjet_mask_ref & lightjet_pileupId_mask & (events.Jet.JES_pt_up >= lightjet_pt_threshold),
            lightjet_mask_ref  & (events.Jet.JES_pt_up >= lightjet_pt_threshold),
        )

        lightjet_JES_down_mask = ak.where(
            (events.Jet.JES_pt_down < 50),
            lightjet_mask_ref & lightjet_pileupId_mask & (events.Jet.JES_pt_down >= lightjet_pt_threshold),
            lightjet_mask_ref  & (events.Jet.JES_pt_down >= lightjet_pt_threshold),
        )

        # JER
        lightjet_JER_up_mask = ak.where(
            (events.Jet.JER_pt_up < 50),
            lightjet_mask_ref & lightjet_pileupId_mask & (events.Jet.JER_pt_up >= lightjet_pt_threshold),
            lightjet_mask_ref  & (events.Jet.JER_pt_up >= lightjet_pt_threshold),
        )

        lightjet_JER_down_mask = ak.where(
            (events.Jet.JER_pt_down < 50),
            lightjet_mask_ref & lightjet_pileupId_mask & (events.Jet.JER_pt_down >= lightjet_pt_threshold),
            lightjet_mask_ref  & (events.Jet.JER_pt_down >= lightjet_pt_threshold),
        )

        return {
            "nominal": lightjet_mask,
            "JES": {
                "up":  lightjet_JES_up_mask, "down":  lightjet_JES_down_mask
            },
            "JER": {
                "up":  lightjet_JER_up_mask, "down":  lightjet_JER_down_mask
            }
        }

    else:
        return  {
            "nominal": lightjet_mask
        }
