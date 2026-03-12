import json
import numpy as np
import awkward as ak
from coffea.nanoevents.methods.base import NanoEventsArray


def select_good_electrons(
    events: NanoEventsArray,
    electron_pt_threshold: int,
    electron_eta_threshold: float,
    electron_id_wp: str,
    electron_iso_wp: str,
    year: str = "2017",
    syst_var: bool = True
) -> ak.Array:
    """
    Build a boolean mask selecting "good" electrons according to
    kinematic, geometric, identification, and isolation requirements.

    Notes
    -----
    - Currently implemented for Run 2 NanoAOD content.
    - The `year` argument is kept for future extensions (e.g. Run 3 IDs).

    Parameters
    ----------
    events : NanoEventsArray
        NanoEvents object containing an Electron collection.

    electron_pt_threshold : int
        Minimum transverse momentum (pT) requirement.

    electron_eta_threshold : float
        Maximum allowed absolute pseudorapidity (|eta|).

    electron_id_wp : str
        Electron identification working point.
        Supported values:
          - MVA IDs: {'wp80iso', 'wp90iso', 'wp80noiso', 'wp90noiso'}
          - Cut-based IDs: {'fail', 'veto', 'loose', 'medium', 'tight'}

    electron_iso_wp : str or None, optional
        Isolation working point {'loose', 'medium', 'tight'}.
        Only used for cut-based IDs or noIso MVA IDs.

    year : str, optional
        Data-taking year (currently unused, reserved for future use).

    Returns
    -------
    ak.Array
        Boolean mask with the same structure as events.Electron,
        selecting electrons that pass all criteria.
    """

    # ===========================================================
    #  Read json file: electron id
    # ============================================================
    # Correction name
    with open("wprime_plus_b/json_files/electron.json", "r") as f:
        electron_info = json.load(f)

    # ============================================================
    # Eta
    # ============================================================     
    # Pseudorapidity acceptance with ECAL barrel–endcap gap removal
    electron_eta_mask = (
        (np.abs(events.Electron.eta) < electron_eta_threshold)
        & (
            (np.abs(events.Electron.eta) < 1.44)
            | (np.abs(events.Electron.eta) > 1.57)
        )
    )

    # ============================================================
    # Combine ID and isolation requirements
    # ============================================================
    if electron_id_wp in ["wp80iso", "wp90iso", "wp80noiso", "wp90noiso"]:
        ID_type = "MVA"
    else:
        ID_type = "cutBased"

    # Iso MVA IDs already include isolation
    if electron_id_wp in ["wp80iso", "wp90iso"]:
        electron_id_iso_mask = getattr(events.Electron, electron_info['Id'][ID_type][electron_id_wp]) 


    else:
        electron_id_iso_mask = (
            id_wps[electron_id_wp]
            & (getattr(events.Electron, electron_info['Iso']['Flag']) < electron_info['Iso'][electron_iso_wp])
        )

    # ============================================================
    # pt
    # ============================================================    
    # Transverse momentum requirement
    electron_pt_mask = (events.Electron.pt >= electron_pt_threshold)
    
    # ============================================================
    # Final electron selection mask
    # ============================================================
    electron_mask_ref = electron_eta_mask & electron_id_iso_mask
    
    electron_mask = electron_pt_mask & electron_mask_ref

    # ============================================================
    # Systematic variations: JES and JEC
    # ============================================================
    # Check if the necessary attributes for JES/JEC variations are present in the events and Electron collection
    if (
        syst_var 
        and hasattr(events, "genWeight")
        and hasattr(events.Electron, "pt_up")
        and hasattr(events.Electron, "pt_down")
    ):
        electron_up_mask = (events.Electron.pt_up >= electron_pt_threshold) & electron_mask_ref
        electron_down_mask = (events.Electron.pt_down >= electron_pt_threshold) & electron_mask_ref
        
        return {
            "nominal": electron_mask,
            "SS": {
                "up": electron_up_mask, "down": electron_down_mask
            }
        }
        
    else:
        return {
            "nominal": electron_mask
        }

