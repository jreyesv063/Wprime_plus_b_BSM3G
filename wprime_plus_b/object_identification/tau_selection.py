import json
import numpy as np
import awkward as ak
from typing import Dict
import importlib.resources
from typing import Optional


def select_good_taus(
    events: ak.Array,
    tau_pt_threshold: float = 20.0,
    tau_eta_threshold: float = 2.4,
    tau_dz_threshold: float = 0.2,
    tau_vs_jet_pass: str = "Tight",
    tau_vs_jet_fail: Optional[float] = None,
    tau_vs_ele: str = "Tight",
    tau_vs_mu: str = "Tight",
    prong: str = "1or3prongs",
    year: str = "2017",
    syst_var: bool  = True
) -> Dict[str, ak.Array]:
    """
    - Run 2: https://twiki.cern.ch/twiki/bin/view/CMS/TauIDRecommendationForRun2#Corrections_to_be_applied_to_gen 
    - Run 3: https://twiki.cern.ch/twiki/bin/view/CMS/TauIDRecommendationForRun3
    Build tau selection masks for different systematic variations.

    Parameters
    ----------
    events : ak.Array
        NanoEvents object containing Tau collection.

    Returns
    -------
    Dict[str, ak.Array]
        Dictionary of boolean masks (per tau) for each variation.
    """

    # ============================================================
    # Load DeepTau working points
    # ============================================================
    with open("wprime_plus_b/json_files/tau.json", "r") as f:
        tau_info = json.load(f)

    tau_id_version = "DeepTau2017" if year in ["2016APV", "2016", "2017", "2018"] else "DeepTau2018"
    suffix = "2017v2p1" if tau_id_version == "DeepTau2017" else "2018v2p5"
    
    # ============================================================
    # Decay mode (prong) selection
    # ============================================================
    tau_dm_mask = ak.zeros_like(events.Tau.decayMode, dtype=bool)
    for decay_mode in tau_info['prongs'][prong]:
        tau_dm_mask = tau_dm_mask | (events.Tau.decayMode == decay_mode)

    # ============================================================
    # ID
    # ============================================================
    tau_id_mask = (
        (getattr(events.Tau, f"idDeepTau{suffix}VSjet") >= tau_info[tau_id_version]["tau_vs_jet"][tau_vs_jet_pass])
        & (getattr(events.Tau, f"idDeepTau{suffix}VSe") >= tau_info[tau_id_version]["tau_vs_e"][tau_vs_ele])
        & (getattr(events.Tau, f"idDeepTau{suffix}VSmu") >= tau_info[tau_id_version]["tau_vs_mu"][tau_vs_mu])
    )
    
    if tau_vs_jet_fail is not None:
        fail_id_mask = getattr(events.Tau, f"idDeepTau{suffix}VSjet") < tau_info[tau_id_version]["tau_vs_jet"][tau_vs_jet_fail]
        tau_id_mask = tau_id_mask & fail_id_mask
        
    # ============================================================
    # Eta
    # ============================================================    
    tau_eta_mask = (np.abs(events.Tau.eta) < tau_eta_threshold)

    # ============================================================
    # dz
    # ============================================================     
    tau_dz_mask = (np.abs(events.Tau.dz) < tau_dz_threshold)


    # ============================================================
    # pt
    # ============================================================       
    tau_pt_mask = (events.Tau.pt >= tau_pt_threshold)


    # ============================================================
    # Final electron selection mask
    # ============================================================        
    tau_mask_ref = tau_dm_mask & tau_id_mask & tau_eta_mask & tau_dz_mask
    
    if year not in ["2016APV", "2016", "2017", "2018"]:
        tau_mask_ref = tau_mask & events.Tau.idDecayModeNewDMs

    tau_mask = tau_mask_ref & tau_pt_mask

    # ============================================================
    # Systematic variations: TES
    # ============================================================      
    # Check if the necessary attributes for TES variations are present in the events and Tau collection
    if (
        syst_var 
        and hasattr(events, "genWeight")
        and hasattr(events.Tau, "pt_up")
        and hasattr(events.Tau, "pt_down")
    ):
        tau_up_mask = (events.Tau.pt_up >= tau_pt_threshold) & tau_mask_ref
        tau_down_mask = (events.Tau.pt_down >= tau_pt_threshold) & tau_mask_ref

        return {
            "nominal": tau_mask,
            "TES": {
                "up": tau_up_mask, "down": tau_down_mask
            }
        }

    else:
        return  {
            "nominal": tau_mask
        }