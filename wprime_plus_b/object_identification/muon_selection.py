import numpy as np
import awkward as ak
from typing import Dict


def select_good_muons(
    events: ak.Array,
    muon_pt_threshold: float,
    muon_eta_threshold: float,
    muon_id_wp: str,
    muon_iso_wp: str,
    is_mc: bool = True,
) -> Dict[str, ak.Array]:
    """
    Build muon selection masks for nominal and systematic variations.

    Parameters
    ----------
    events : ak.Array
        NanoEvents object containing Muon collection.

    Returns
    -------
    Dict[str, ak.Array]
        Dictionary of boolean masks (per muon) for each variation.
    """

    # ============================================================
    # Validate working points
    # ============================================================

    valid_id_wps = {"Highpt", "Loose", "Medium", "Tight"}
    valid_iso_wps = {"Loose", "Medium", "Tight"}

    if muon_id_wp not in valid_id_wps:
        raise ValueError(f"Invalid muon_id_wp: {muon_id_wp}")

    if muon_iso_wp not in valid_iso_wps:
        raise ValueError(f"Invalid muon_iso_wp: {muon_iso_wp}")

    # ============================================================
    # Kinematic masks (eta independent of pt shifts)
    # ============================================================

    muon_eta_mask = np.abs(events.Muon.eta) < muon_eta_threshold

    # ============================================================
    # Muon ID masks
    # ============================================================
    id_wps = {
        "Highpt": events.Muon.highPtId == 2,
        "Loose": events.Muon.looseId,
        "Medium": events.Muon.mediumId,
        "Tight": events.Muon.tightId,
    }

    muon_id_mask = id_wps[muon_id_wp]

    # ============================================================
    # Muon isolation masks
    # ============================================================

    if hasattr(events.Muon, "pfRelIso04_all"):
        rel_iso = events.Muon.pfRelIso04_all
    else:
        rel_iso = events.Muon.pfRelIso03_all

    iso_wps = {
        "Loose": rel_iso < 0.25,
        "Medium": rel_iso < 0.20,
        "Tight": rel_iso < 0.15,
    }

    muon_iso_mask = iso_wps[muon_iso_wp]

    # ============================================================
    # Build masks for pt variations
    # ============================================================

    good_muon_masks = {}

    if is_mc and hasattr(events.Muon, "pt_up"):
        pt_variations = {
            "nominal": events.Muon.pt,
            "up": events.Muon.pt_up,
            "down": events.Muon.pt_down,
        }
    else:
        pt_variations = {
            "nominal": events.Muon.pt
        }

    for name, muon_pt in pt_variations.items():
        muon_pt_mask = muon_pt >= muon_pt_threshold

        good_muon_masks[name] = (
            muon_pt_mask
            & muon_eta_mask
            & muon_id_mask
            & muon_iso_mask
        )

    return good_muon_masks
