import json
import numpy as np
import awkward as ak
from typing import Dict
import importlib.resources



def select_good_fatjets(
    fatjets,
    year: str = "2017",
    fatjet_pt_threshold: float = 300.0,
    fatjet_eta_threshold: float = 2.4,
    TvsQCD: str = "Tight",
    is_mc: bool = True,
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
    # Wps top tagger, jet_id and jet_eta
    with open("wprime_plus_b/json_files/topWps.json", "r") as f:
        Wps = json.load(f)
    #with open("wprime_plus_b/jsons/topWps.json", "r") as f: 
    #    Wps = json.load(f)

    pNet_id = Wps[year]["TvsQCD"][TvsQCD]
    jet_id = Wps[year]["jet_id"]

    good_fatjet_masks: Dict[str, ak.Array] = {}

    # ------------------------------------------------------------------
    # Helper function: base fatjet selection
    # ------------------------------------------------------------------
    def fatjet_selection(jets: ak.Array) -> ak.Array:
        """
        Apply baseline fatjet selection.
        """
        return (
            (jets.pt >= fatjet_pt_threshold)
            & (np.abs(jets.eta) <= fatjet_eta_threshold)
            & (jets.particleNet_TvsQCD >= pNet_id)
            & (jets.jetId >= jet_id)
        )

    # ------------------------------------------------------------------
    # MC samples
    # ------------------------------------------------------------------
    if is_mc:

        # If JES/JER information is missing, fall back to nominal only
        if "JES_jes" not in fatjets.fields or "JER" not in fatjets.fields:
            good_fatjet_masks["nominal"] = fatjet_selection(fatjets)

            # Fill missing systematics with empty masks for consistency
            empty_mask = ak.zeros_like(fatjets.pt, dtype=bool)
            good_fatjet_masks["JES_up"] = empty_mask
            good_fatjet_masks["JES_down"] = empty_mask
            good_fatjet_masks["JER_up"] = empty_mask
            good_fatjet_masks["JER_down"] = empty_mask

            return good_fatjet_masks

        # Define systematic variations
        jet_shifts = {
            "nominal": fatjets,
            "JES_up": fatjets.JES_jes.up,
            "JES_down": fatjets.JES_jes.down,
            "JER_up": fatjets.JER.up,
            "JER_down": fatjets.JER.down,
        }

        # Apply selection to each variation
        for name, jets in jet_shifts.items():
            good_fatjet_masks[name] = fatjet_selection(jets)

    # ------------------------------------------------------------------
    # Data samples (no systematics)
    # ------------------------------------------------------------------
    else:
        good_fatjet_masks["nominal"] = fatjet_selection(fatjets)

    return good_fatjet_masks
