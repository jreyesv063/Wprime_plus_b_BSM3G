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
    tau_vs_jet_pass: str = "T",
    tau_vs_jet_fail: Optional[float] = None,
    tau_vs_ele: str = "T",
    tau_vs_mu: str = "T",
    prong: int = 13,
    is_mc: bool = True,
    year: str = "2017",
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
    with open("wprime_plus_b/json_files/tau_wps.json", "r") as f:
        tau_wps_all = json.load(f)
        
    # ============================================================
    # Decay mode (prong) selection
    # ============================================================

    prong_to_modes = {
        1: [0, 1, 2],           # 1 prong
        2: [5, 6, 7],           # 2 prongs 
        3: [10, 11],            # 3 prongs 
        12: [0, 1, 2, 5, 6, 7], # 1 or 2 prongs
        13: [0, 1, 2, 10, 11],  # 1 or 3 prongs
        23: [5, 6, 7, 10, 11],  # 2 or 3 prongs
    }

    if prong not in prong_to_modes:
        raise ValueError(
            f"Invalid prong={prong}. Allowed values: {sorted(prong_to_modes)}"
        )

    tau_dm = events.Tau.decayMode
    decay_mode_mask = ak.zeros_like(tau_dm, dtype=bool)

    for mode in prong_to_modes[prong]:
        decay_mode_mask = decay_mode_mask | (tau_dm == mode)

    # ============================================================
    # Helper to build tau mask
    # ============================================================
    def build_tau_mask(tau_pt: ak.Array) -> ak.Array:
        
        tau_id_version = "DeepTau2017" if year in ["2016APV", "2016", "2017", "2018"] else "DeepTau2018"
        suffix = "2017v2p1" if tau_id_version == "DeepTau2017" else "2018v2p5"

        # Load working points
        tau_wps = tau_wps_all[tau_id_version]

        tau_mask = (
            (tau_pt > tau_pt_threshold)
            & (np.abs(events.Tau.eta) < tau_eta_threshold)
            & (np.abs(events.Tau.dz) < tau_dz_threshold)
            & (getattr(events.Tau, f"idDeepTau{suffix}VSjet") > tau_wps["deep_tau_jet"][tau_vs_jet_pass])
            & (getattr(events.Tau, f"idDeepTau{suffix}VSe") > tau_wps["deep_tau_electron"][tau_vs_ele])
            & (getattr(events.Tau, f"idDeepTau{suffix}VSmu") > tau_wps["deep_tau_muon"][tau_vs_mu])
            & decay_mode_mask
        )

        if tau_id_version == "DeepTau2018":
            tau_mask = tau_mask & events.Tau.idDecayModeNewDMs

        if tau_vs_jet_fail is not None:
            fail_tau_id = getattr(events.Tau, f"idDeepTau{suffix}VSjet") < tau_wps["deep_tau_jet"][tau_vs_jet_fail]
            tau_mask = tau_mask & fail_tau_id


        return tau_mask

    # ============================================================
    # Build masks for nominal / systematics
    # ============================================================

    good_tau_masks = {}

    if is_mc:
        pt_variations = {
            "nominal": events.Tau.pt,
            "up": events.Tau.pt_up,
            "down": events.Tau.pt_down,
        }

        for name, tau_pt in pt_variations.items():
            good_tau_masks[name] = build_tau_mask(tau_pt)

    else:
        good_tau_masks["nominal"] = build_tau_mask(events.Tau.pt)

    return good_tau_masks
