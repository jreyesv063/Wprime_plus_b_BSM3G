import json
import numpy as np
import awkward as ak
import importlib.resources
from pathlib import Path
from coffea.nanoevents.methods.base import NanoEventsArray


def select_good_muons(
    events: ak.Array, muon_pt_threshold: int, muon_eta_threshold: float, muon_id_wp: str, muon_iso_wp: str, year: str
) -> ak.highlevel.Array:
    """
    return mask from a collection of events based on specified criteria.

    Parameters:
    -----------
    events:
        A collection of events represented using the NanoEventsArray class.

    muon_pt_threshold:
        Muon transverse momentum threshold

    muon_id_wp:
        Muon ID working point. Available working points for the CutBased ID {'loose', 'medium', 'tight'}

    muon_iso_wp:
        Muon ISO working point {'loose', 'medium', 'tight'}

    Returns:
    --------
        An Awkward Array mask containing the selected "good" muons that satisfy the specified criteria.
    """
    # muon pT threshold. Conditional depending on the year. For 2016, 2016APV, and 2018, the threshold is 27 GeV. For 2017, it is 30 GeV.
    if year in ["2016APV", "2016", "2018"]:
        muon_pt_mask = events.Muon.pt >= 27.0
    else:
        muon_pt_mask = events.Muon.pt >= muon_pt_threshold
    # electron pseudorapidity mask
    muon_eta_mask = np.abs(events.Muon.eta) < muon_eta_threshold

    # muon ID mask https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedElectronIdentificationRun2
    id_wps = {
        "highpt": events.Muon.highPtId == 2,
        # cutbased ID working points
        "loose": events.Muon.looseId,
        "medium": events.Muon.mediumId,
        "tight": events.Muon.tightId,
    }
    muon_id_mask = id_wps[muon_id_wp]

    # muon ID and Iso mask https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonSelection
    iso_wps = {
        "loose": (
            events.Muon.pfRelIso04_all < 0.25
            if hasattr(events.Muon, "pfRelIso04_all")
            else events.Muon.pfRelIso03_all < 0.25
        ),
        "medium": (
            events.Muon.pfRelIso04_all < 0.20
            if hasattr(events.Muon, "pfRelIso04_all")
            else events.Muon.pfRelIso03_all < 0.20
        ),
        "tight": (
            events.Muon.pfRelIso04_all < 0.15
            if hasattr(events.Muon, "pfRelIso04_all")
            else events.Muon.pfRelIso03_all < 0.15
        ),
    }
    muon_iso_mask = iso_wps[muon_iso_wp]

    return (muon_pt_mask) & (muon_eta_mask) & (muon_id_mask) & (muon_iso_mask)