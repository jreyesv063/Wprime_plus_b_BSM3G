import json
import numpy as np
import awkward as ak
from typing import Dict
import importlib.resources as resources



def select_good_wjets(
    wjets: ak.Array,
    year: str = "2017",
    w_pt_threshold: float = 200.0,
    w_eta_threshold: float = 2.4,
    WvsQCD: str = "Tight",
    is_mc: bool = True,
) -> Dict[str, ak.Array]:
    """
    Build masks selecting good W-tagged jets for nominal and JES/JER variations.

    Parameters
    ----------
    wjets : ak.Array
        W jet collection.
    year : str
        Data-taking year (used to read working points).
    w_pt_threshold : float
        Minimum pT requirement for W jets.
    w_eta_threshold : float
        Maximum |eta| requirement for W jets.
    WvsQCD : str
        ParticleNet WvsQCD working point.
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
    # Wps top tagger, jet_id and jet_eta
    with open("wprime_plus_b/json_files/topWps.json", "r") as f:
        Wps = json.load(f)

    #with open("wprime_plus_b/jsons/topWps.json", "r") as f: 
    #    Wps = json.load(f)

    pNet_id = Wps[year]["WvsQCD"][WvsQCD]
    jet_id = Wps[year]["jet_id"]

    good_wjet_masks: Dict[str, ak.Array] = {}

    # ------------------------------------------------------------------
    # Helper function: base W jet selection
    # ------------------------------------------------------------------
    def wjet_selection(jets: ak.Array) -> ak.Array:
        """
        Apply baseline W jet selection.
        """
        return (
            (jets.pt >= w_pt_threshold)
            & (np.abs(jets.eta) <= w_eta_threshold)
            & (jets.particleNet_WvsQCD >= pNet_id)
            & (jets.jetId >= jet_id)
        )

    # ------------------------------------------------------------------
    # MC samples
    # ------------------------------------------------------------------
    if is_mc:

        # If JES/JER information is missing, fall back to nominal only
        if "JES_jes" not in wjets.fields or "JER" not in wjets.fields:
            good_wjet_masks["nominal"] = wjet_selection(wjets)

            # Fill missing systematics with empty masks for consistency
            empty_mask = ak.zeros_like(wjets.pt, dtype=bool)
            good_wjet_masks["JES_up"] = empty_mask
            good_wjet_masks["JES_down"] = empty_mask
            good_wjet_masks["JER_up"] = empty_mask
            good_wjet_masks["JER_down"] = empty_mask

            return good_wjet_masks

        # Define systematic variations
        jet_shifts = {
            "nominal": wjets,
            "JES_up": wjets.JES_jes.up,
            "JES_down": wjets.JES_jes.down,
            "JER_up": wjets.JER.up,
            "JER_down": wjets.JER.down,
        }

        # Apply selection to each variation
        for name, jets in jet_shifts.items():
            good_wjet_masks[name] = wjet_selection(jets)

    # ------------------------------------------------------------------
    # Data samples (no systematics)
    # ------------------------------------------------------------------
    else:
        good_wjet_masks["nominal"] = wjet_selection(wjets)

    return good_wjet_masks
