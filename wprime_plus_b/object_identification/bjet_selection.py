import json
import numpy as np
import awkward as ak
from typing import Dict
import importlib.resources
from typing import Optional


def select_good_bjets(
    events: ak.Array,
    year: str = "2017",
    btag_working_point_pass: str = "Medium",
    btag_working_point_fail: str = "", # can be empty
    bjet_pt_threshold: int = 20,
    bjet_eta_threshold: float = 2.4,
    bjet_id_wp: str = "TightLepVeto",
    bjet_pileup_id: str = "Tight",
    syst_var: bool = True
) -> Dict[str, ak.Array]:
    
    # ------------------------------------------------------------------
    # Load working points from JSON
    # ------------------------------------------------------------------
    with open("wprime_plus_b/json_files/jet.json", "r") as f:
        bjet_info = json.load(f)

    # --------------------------------------------------
    # Load b-tag thresholds
    # --------------------------------------------------
    with open("wprime_plus_b/json_files/bjet.json", "r") as f:
        btag_thresholds = json.load(f)["deepJet"][year]
        
        btag_pass = btag_thresholds[btag_working_point_pass]
        # optional: btag_fail
        btag_fail = (
            btag_thresholds[btag_working_point_fail]
            if btag_working_point_fail
            else None
        )

    # ============================================================
    # Eta
    # ============================================================
    bjet_eta_mask = np.abs(events.Jet.eta) < bjet_eta_threshold   

    # ============================================================
    # jet ID
    # ============================================================
    if year in ["2016APV", "2016", "2017", "2018"]:
        bjet_jetId_mask = (events.Jet.jetId  == bjet_info["jetid"][year][bjet_id_wp])


    # ============================================================
    # Btag flavor
    # ============================================================
    bjet_btag_mask = (events.Jet.btagDeepFlavB >= btag_pass)      # DeepJet b+bb+lepb tag discriminator

    if btag_fail is not None:
        bjet_btag_mask = bjet_btag_mask & (events.Jet.btagDeepFlavB < btag_fail)
    
    # ============================================================
    # pt
    # ============================================================
    bjet_pt_mask = (events.Jet.pt >= bjet_pt_threshold)

    
    # ============================================================
    # pileupjet ID (pt < 50)
    # ============================================================
    bjet_pileupId_mask = (events.Jet.puId  == bjet_info["pujetid"][year][bjet_pileup_id])  
    
    
    # ============================================================
    # Final bjet selection mask
    # ============================================================ 
    bjet_mask_ref = bjet_eta_mask & bjet_jetId_mask & bjet_btag_mask


    bjet_mask = ak.where(
        (events.Jet.pt < 50),
        bjet_mask_ref & bjet_pileupId_mask & bjet_pt_mask,
        bjet_mask_ref  & bjet_pt_mask,
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
        bjet_JES_up_mask = ak.where(
            (events.Jet.JES_pt_up < 50),
            bjet_mask_ref & bjet_pileupId_mask & (events.Jet.JES_pt_up >= bjet_pt_threshold),
            bjet_mask_ref  & (events.Jet.JES_pt_up >= bjet_pt_threshold),
        )

        bjet_JES_down_mask = ak.where(
            (events.Jet.JES_pt_down < 50),
            bjet_mask_ref & bjet_pileupId_mask & (events.Jet.JES_pt_down >= bjet_pt_threshold),
            bjet_mask_ref  & (events.Jet.JES_pt_down >= bjet_pt_threshold),
        )

        # JER
        bjet_JER_up_mask = ak.where(
            (events.Jet.JER_pt_up < 50),
            bjet_mask_ref & bjet_pileupId_mask & (events.Jet.JER_pt_up >= bjet_pt_threshold),
            bjet_mask_ref  & (events.Jet.JER_pt_up >= bjet_pt_threshold),
        )

        bjet_JER_down_mask = ak.where(
            (events.Jet.JER_pt_down < 50),
            bjet_mask_ref & bjet_pileupId_mask & (events.Jet.JER_pt_down >= bjet_pt_threshold),
            bjet_mask_ref  & (events.Jet.JER_pt_down >= bjet_pt_threshold),
        )

        return {
            "nominal": bjet_mask,
            "JES": {
                "up":  bjet_JES_up_mask, "down":  bjet_JES_down_mask
            },
            "JER": {
                "up":  bjet_JER_up_mask, "down":  bjet_JER_down_mask
            }
        }

    else:
        return  {
            "nominal": bjet_mask
        }
