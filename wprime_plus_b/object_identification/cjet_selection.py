import json
import numpy as np
import awkward as ak
from typing import Dict
import importlib.resources
from typing import Optional


def select_good_cjets(
    events: ak.Array,
    year: str = "2017",
    ctag_working_point_pass: str = "Tight",
    cjet_pt_threshold: int = 20,
    cjet_eta_threshold: float = 2.4,
    cjet_id_wp: str = "TightLepVeto",
    cjet_pileup_id: str = "Tight",
    syst_var: bool = True
) -> Dict[str, ak.Array]:
    
    # ------------------------------------------------------------------
    # Load working points from JSON
    # ------------------------------------------------------------------
    with open("wprime_plus_b/json_files/jet.json", "r") as f:
        cjet_info = json.load(f)

    # --------------------------------------------------
    # Load b-tag thresholds
    # --------------------------------------------------
    with open("wprime_plus_b/json_files/cjet.json", "r") as f:
        ctag_thresholds = json.load(f)["deepJet"][year]

        ctag_threshold = [
            ctag_thresholds["CvB_cut"][ctag_working_point_pass], 
            ctag_thresholds["CvL_cut"][ctag_working_point_pass]
        ]

    # ============================================================
    # Eta
    # ============================================================
    cjet_eta_mask = np.abs(events.Jet.eta) < cjet_eta_threshold   

    # ============================================================
    # jet ID
    # ============================================================
    if year in ["2016APV", "2016", "2017", "2018"]:
        cjet_jetId_mask = (events.Jet.jetId  == cjet_info["jetid"][year][cjet_id_wp])


    # ============================================================
    # Ctag flavor
    # ============================================================
    cjet_btag_mask = (
        (events.Jet.btagDeepFlavCvB > ctag_threshold[0])      # DeepJet c vs b+bb+lepb discriminator
        & (events.Jet.btagDeepFlavCvL > ctag_threshold[1])    # DeepJet c vs uds+g discriminator
    )

    
    # ============================================================
    # pt
    # ============================================================
    cjet_pt_mask = (events.Jet.pt >= cjet_pt_threshold)

    
    # ============================================================
    # pileupjet ID (pt < 50)
    # ============================================================
    cjet_pileupId_mask = (events.Jet.puId  == cjet_info["jetid"][year][cjet_pileup_id])  
    
    
    # ============================================================
    # Final bjet selection mask
    # ============================================================ 
    cjet_mask_ref = cjet_eta_mask & cjet_jetId_mask & cjet_btag_mask
        

    cjet_mask = ak.where(
        (events.Jet.pt < 50),
        cjet_mask_ref & cjet_pileupId_mask & cjet_pt_mask,
        cjet_mask_ref  & cjet_pt_mask,
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
        cjet_JES_up_mask = ak.where(
            (events.Jet.JES_pt_up < 50),
            cjet_mask_ref & cjet_pileupId_mask & (events.Jet.JES_pt_up >= cjet_pt_threshold),
            cjet_mask_ref  & (events.Jet.JES_pt_up >= cjet_pt_threshold),
        )

        cjet_JES_down_mask = ak.where(
            (events.Jet.JES_pt_down < 50),
            cjet_mask_ref & cjet_pileupId_mask & (events.Jet.JES_pt_down >= cjet_pt_threshold),
            cjet_mask_ref  & (events.Jet.JES_pt_down >= cjet_pt_threshold),
        )

        # JER
        cjet_JER_up_mask = ak.where(
            (events.Jet.JER_pt_up < 50),
            cjet_mask_ref & cjet_pileupId_mask & (events.Jet.JER_pt_up >= cjet_pt_threshold),
            cjet_mask_ref  & (events.Jet.JER_pt_up >= cjet_pt_threshold),
        )

        cjet_JER_down_mask = ak.where(
            (events.Jet.JES_pt_down < 50),
            cjet_mask_ref & cjet_pileupId_mask & (events.Jet.JER_pt_down >= cjet_pt_threshold),
            cjet_mask_ref  & (events.Jet.JER_pt_down >= cjet_pt_threshold),
        )

        return {
            "nominal": cjet_mask,
            "JES": {
                "up":  cjet_JES_up_mask, "down":  cjet_JES_down_mask
            },
            "JER": {
                "up":  cjet_JER_up_mask, "down":  cjet_JER_down_mask
            }
        }

    else:
        return  {
            "nominal": cjet_mask
        }