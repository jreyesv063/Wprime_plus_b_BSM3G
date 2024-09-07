import json
import numpy as np
import awkward as ak
import importlib.resources
from pathlib import Path
from coffea.nanoevents.methods.base import NanoEventsArray

def select_good_electrons(
    events: NanoEventsArray,
    electron_pt_threshold: int,
    electron_id_wp: str,
    electron_iso_wp: str = None,
) -> ak.highlevel.Array:
    """
    Selects and filters "good" electrons from a collection of events based on specified criteria.

    Parameters:
    -----------
    events:
        A collection of events represented using the NanoEventsArray class.

    electron_pt_threshold:
        Electron transverse momentum threshold

    electron_id_wp:
        Electron ID working point. Available working point for the CutBased and the MVA IDs.
        MVA: {'wp80iso', 'wp90iso', 'wp80noiso', 'wp90noiso'}
        CutBased: {'loose', 'medium', 'tight'}

    electron_iso_wp:
        Electron ISO working point {'loose', 'medium', 'tight'}. Only used for CutBased IDs or noIso MVA IDs

    Returns:
    --------
        An Awkward Array mask containing the selected "good" electrons that satisfy the specified criteria.
    """
    # electron pT threshold
    electron_pt_mask = events.Electron.pt >= electron_pt_threshold
    # electron pseudorapidity mask
    electron_eta_mask = (np.abs(events.Electron.eta) < 2.4) & (
        (np.abs(events.Electron.eta) < 1.44) | (np.abs(events.Electron.eta) > 1.57)
    )
    # electron ID and Iso masks
    id_wps = {
        # mva ID working points https://twiki.cern.ch/twiki/bin/view/CMS/MultivariateElectronIdentificationRun2
        "wp80iso": events.Electron.mvaFall17V2Iso_WP80,
        "wp90iso": events.Electron.mvaFall17V2Iso_WP90,
        "wp80noiso": events.Electron.mvaFall17V2noIso_WP80,
        "wp90noiso": events.Electron.mvaFall17V2noIso_WP90,
        # cutbased ID working points https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedElectronIdentificationRun2
        "loose": events.Electron.cutBased == 2,
        "medium": events.Electron.cutBased == 3,
        "tight": events.Electron.cutBased == 4,
    }
    iso_wps = {
        # https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonSelection
        "loose": events.Electron.pfRelIso04_all < 0.25
        if hasattr(events.Electron, "pfRelIso04_all")
        else events.Electron.pfRelIso03_all < 0.25,
        "medium": events.Electron.pfRelIso04_all < 0.20
        if hasattr(events.Electron, "pfRelIso04_all")
        else events.Electron.pfRelIso03_all < 0.20,
        "tight": events.Electron.pfRelIso04_all < 0.15
        if hasattr(events.Electron, "pfRelIso04_all")
        else events.Electron.pfRelIso03_all < 0.15,
    }
    if electron_id_wp in ["wp80iso", "wp90iso"]:
        electron_id_iso_mask = id_wps[electron_id_wp]
    else:
        electron_id_iso_mask = (id_wps[electron_id_wp]) & (iso_wps[electron_iso_wp])

    return electron_pt_mask & electron_eta_mask & electron_id_iso_mask