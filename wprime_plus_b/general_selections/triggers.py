import json
import numpy as np
import awkward as ak
import importlib.resources
from wprime_plus_b.processors.utils.analysis_utils import trigger_match


def get_trigger_mask(
    events: ak.Array,
    lepton_flavor: str,
    year: str,
    reference_trigger: str,
    muon_id: str = None,
    electron_id: str = None,
):
    """
    Build a boolean mask for events that pass the reference trigger(s).

    If multiple triggers share the same root name (e.g., due to different versions),
    the function performs a logical OR across all of them. An event passes the mask
    if it passes **any** of the selected triggers.

    Parameters
    ----------
    events : ak.Array
        NanoEvents array containing HLT information (events.HLT).
    lepton_flavor : str
        One of {"mu", "ele", "tau"}.
    year : str
        Data-taking year, e.g., "2017", "2018".
    reference_trigger : str
        Reference trigger name in the JSON configuration.
    muon_id : str, optional
        Working point of muon ID (used if lepton_flavor == "mu").
    electron_id : str, optional
        Working point of electron ID (used if lepton_flavor == "ele").

    Returns
    -------
    mask_reference_trigger : ak.Array
        Boolean array per event: True if event passes any of the reference triggers.
    reference_triggers : list
        List of trigger names used to build the mask.
    """
    # Load triggers from JSON
    with open("wprime_plus_b/json_files/triggers.json", "r") as f:
        triggers_json = json.load(f) 

    # Determine reference triggers
    if lepton_flavor == "mu":
        if muon_id is None:
            raise ValueError("muon_id must be provided for muon triggers")
        reference_triggers = triggers_json[year][reference_trigger][muon_id]

    elif lepton_flavor == "ele":
        if electron_id is None:
            raise ValueError("electron_id must be provided for electron triggers")
        reference_triggers = triggers_json[year][reference_trigger][electron_id]

    elif lepton_flavor == "tau":
        ref_trigger_list = triggers_json[year][reference_trigger]
        # Only include triggers present in events
        reference_triggers = [
            trig for trig in events.HLT.fields if any(trig.startswith(r) for r in ref_trigger_list)
        ]
    else:
        raise ValueError(f"Unknown lepton flavor '{lepton_flavor}'")

    # Combine masks for all reference triggers using awkward's broadcasting
    masks = [events.HLT[trig] for trig in reference_triggers if trig in events.HLT.fields]
    if masks:
        mask_reference_trigger = ak.any(ak.Array(masks), axis=0)
    else:
        # If no matching triggers, return all False
        mask_reference_trigger = ak.zeros(len(events), dtype=bool)

    return mask_reference_trigger, reference_triggers




def get_trigger_match_mask(
    events: ak.Array,
    leptons: dict,
    lepton_flavor: str,
    year: str,
    electron_id_wp: str,
    muon_id_wp: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a trigger mask and DeltaR-matched trigger object mask for events.

    For electrons and muons, the function selects events passing the reference triggers
    and ensures at least one lepton matches a trigger object (TrigObj) using `trigger_match`.
    For taus, all events are considered to pass the trigger match.

    Parameters
    ----------
    events : ak.Array
        NanoEvents array containing HLT information (events.HLT) and trigger objects (events.TrigObj).
    leptons : dict
        Dictionary with lepton collections, e.g., {"ele": electrons, "mu": muons}.
    lepton_flavor : str
        Lepton flavor to process ("ele", "mu", or "tau").
    year : str
        Data-taking year, e.g., "2017", "2018".
    electron_id_wp : str
        Electron ID working point used for trigger selection.
    muon_id_wp : str
        Muon ID working point used for trigger selection.

    Returns
    -------
    trigger_mask : np.ndarray
        Boolean array per event. True if the event passes any reference trigger.
    trigger_match_mask : np.ndarray
        Boolean array per event. True if the event has at least one lepton matched to a trigger object.
    """
    nevents = len(events)

    if lepton_flavor == "tau":
        # For taus, assume all events pass
        return np.ones(nevents, dtype=bool), np.ones(nevents, dtype=bool)

    # Load trigger paths from JSON
    with importlib.resources.path("wprime_plus_b.data", "triggers.json") as path:
        with open(path, "r") as handle:
            triggers = json.load(handle)[year][lepton_flavor]

    id_wp = electron_id_wp if lepton_flavor == "ele" else muon_id_wp
    trigger_paths = triggers[id_wp]

    # Initialize masks
    trigger_mask = np.zeros(nevents, dtype=bool)
    trigger_match_mask = np.zeros(nevents, dtype=bool)

    # Loop once over all trigger paths
    for tp in trigger_paths:
        if tp in events.HLT.fields:
            trigger_mask |= events.HLT[tp]
            trig_match_mask = trigger_match(
                leptons=leptons[lepton_flavor],
                trigobjs=events.TrigObj,
                trigger_path=tp
            )
            trigger_match_mask |= trig_match_mask

    return trigger_mask, trigger_match_mask