import json
import numpy as np
import awkward as ak
from typing import Dict
import importlib.resources
from typing import Optional


def select_good_jets(
    events: ak.Array,
    year: str = "2017",
    jet_pt_threshold: int = 20,
    jet_eta_threshold: float = 2.4,
    jet_id_wp: str = "TightLepVeto",
    jet_pileup_id: str = "Tight",
    syst_var: bool = True
) -> Dict[str, ak.Array]:
    
    # ------------------------------------------------------------------
    # Load working points from JSON
    # ------------------------------------------------------------------
    with open("wprime_plus_b/json_files/jet.json", "r") as f:
        jet_info = json.load(f)


    # ============================================================
    # Eta
    # ============================================================
    jet_eta_mask = np.abs(events.Jet.eta) < jet_eta_threshold   

    # ============================================================
    # jet ID
    # ============================================================
    if year in ["2016APV", "2016", "2017", "2018"]:
        jet_jetId_mask = (events.Jet.jetId  == jet_info["jetid"][year][jet_id_wp])

    
    # ============================================================
    # pt
    # ============================================================
    jet_pt_mask = (events.Jet.pt >= jet_pt_threshold)

    
    # ============================================================
    # pileupjet ID (pt < 50)
    # ============================================================
    jet_pileupId_mask = (events.Jet.puId  == jet_info["pujetid"][year][jet_pileup_id])  
    
    
    # ============================================================
    # Final Jet selection mask
    # ============================================================ 
    jet_mask_ref = jet_eta_mask & jet_jetId_mask 


    jet_mask = ak.where(
        (events.Jet.pt < 50),
        jet_mask_ref & jet_pileupId_mask & jet_pt_mask,
        jet_mask_ref & jet_pt_mask,
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
        jet_JES_up_mask = ak.where(
            (events.Jet.JES_pt_up < 50),
            jet_mask_ref & jet_pileupId_mask & (events.Jet.JES_pt_up >= jet_pt_threshold),
            jet_mask_ref  & (events.Jet.JES_pt_up >= jet_pt_threshold),
        )

        jet_JES_down_mask = ak.where(
            (events.Jet.JES_pt_down < 50),
            jet_mask_ref & jet_pileupId_mask & (events.Jet.JES_pt_down >= jet_pt_threshold),
            jet_mask_ref  & (events.Jet.JES_pt_down >= jet_pt_threshold),
        )

        # JER
        jet_JER_up_mask = ak.where(
            (events.Jet.JER_pt_up < 50),
            jet_mask_ref & jet_pileupId_mask & (events.Jet.JER_pt_up >= jet_pt_threshold),
            jet_mask_ref  & (events.Jet.JER_pt_up >= jet_pt_threshold),
        )

        jet_JER_down_mask = ak.where(
            (events.Jet.JER_pt_down < 50),
            jet_mask_ref & jet_pileupId_mask & (events.Jet.JER_pt_down >= jet_pt_threshold),
            jet_mask_ref  & (events.Jet.JER_pt_down >= jet_pt_threshold),
        )

        return {
            "nominal": jet_mask,
            "JES": {
                "up":  jet_JES_up_mask, "down":  jet_JES_down_mask
            },
            "JER": {
                "up":  jet_JER_up_mask, "down":  jet_JER_down_mask
            }
        }

    else:
        return  {
            "nominal": jet_mask
        }