import json
import numpy as np
import awkward as ak
import importlib.resources
from pathlib import Path
from coffea.nanoevents.methods.base import NanoEventsArray


def select_good_taus(
    events: ak.Array,
    tau_pt_threshold: float,
    tau_eta_threshold: float,
    tau_dz_threshold: float,
    tau_vs_jet: str,
    tau_vs_ele: str,
    tau_vs_mu: str,
    prong: int,
) -> ak.highlevel.Array:
    """
    Selects and filters "good" taus from a collection of events based on specified criteria.

    Parameters:
    -----------
    events:
        A collection of events represented using the NanoEventsArray class.

    Returns:
    --------
        An Awkward Array mask containing the selected "good" taus that satisfy the specified criteria.
    """
    with importlib.resources.open_text("wprime_plus_b.data", "tau_wps.json") as file:
        taus_wps = json.load(file)
    # The DM is defined using the "new" decay mode reconstruction, binned in DMs 0, 1, 10, and 11.
    # https://github.com/uhh-cms/hh2bbtautau/blob/78fb359ea275e4eb2bc4cafbe238efa052d6355f/hbt/production/tau.py#L83

    prong_to_modes = {
        1: [0, 1, 2],
        2: [5, 6, 7],
        3: [10, 11],
        13: [0, 1, 2, 10, 11],
        12: [0, 1, 2, 5, 6, 7],
        23: [5, 6, 7, 10, 11],
    }
    if prong not in prong_to_modes:
        raise ValueError(
            "Invalid prong value. Please specify 1, 2, 3, 12, 13 or 23 for the prong parameter."
        )
    tau_dm = events.Tau.decayMode
    decay_mode_mask = ak.zeros_like(tau_dm)
    for mode in prong_to_modes[prong]:
        decay_mode_mask = np.logical_or(decay_mode_mask, tau_dm == mode)
        
    # Pasa Loose y falla tight
    good_taus = (
        (events.Tau.pt > tau_pt_threshold)
        & (np.abs(events.Tau.eta) < tau_eta_threshold)
        & (np.abs(events.Tau.dz) < tau_dz_threshold)
        & (
            events.Tau.idDeepTau2017v2p1VSjet
            > taus_wps["DeepTau2017"]["deep_tau_jet"][tau_vs_jet]
        )
        & (
            events.Tau.idDeepTau2017v2p1VSe
            > taus_wps["DeepTau2017"]["deep_tau_electron"][tau_vs_ele]
        )
        & (
            events.Tau.idDeepTau2017v2p1VSmu
            > taus_wps["DeepTau2017"]["deep_tau_muon"][tau_vs_mu]
        )
        & (decay_mode_mask)
    )
    return good_taus