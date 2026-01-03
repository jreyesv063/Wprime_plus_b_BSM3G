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

    # ============================================================
    # Kinematic selection
    # ============================================================

    # Transverse momentum requirement
    electron_pt_mask = events.Electron.pt >= electron_pt_threshold

    # Pseudorapidity acceptance with ECAL barrel–endcap gap removal
    electron_eta_mask = (
        (np.abs(events.Electron.eta) < electron_eta_threshold)
        & (
            (np.abs(events.Electron.eta) < 1.44)
            | (np.abs(events.Electron.eta) > 1.57)
        )
    )

    # ============================================================
    # Electron identification working points
    # ============================================================

    id_wps = {
        # MVA-based electron IDs (Run 2): https://twiki.cern.ch/twiki/bin/view/CMS/MultivariateElectronIdentificationRun2
        "wp80iso": events.Electron.mvaFall17V2Iso_WP80,
        "wp90iso": events.Electron.mvaFall17V2Iso_WP90,
        "wp80noiso": events.Electron.mvaFall17V2noIso_WP80,
        "wp90noiso": events.Electron.mvaFall17V2noIso_WP90,

        # Cut-based electron IDs (Run 2): https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedElectronIdentificationRun2
        "Fail": events.Electron.cutBased == 0,
        "Veto": events.Electron.cutBased == 1,
        "Loose": events.Electron.cutBased == 2,
        "Medium": events.Electron.cutBased == 3,
        "Tight": events.Electron.cutBased == 4,
    }

    # ============================================================
    # Isolation working points
    # ============================================================

    # Prefer pfRelIso04 if available, otherwise fall back to pfRelIso03
    if hasattr(events.Electron, "pfRelIso04_all"):
        rel_iso = events.Electron.pfRelIso04_all
    else:
        rel_iso = events.Electron.pfRelIso03_all

    iso_wps = {
        "Loose": rel_iso < 0.25,
        "Medium": rel_iso < 0.20,
        "Tight": rel_iso < 0.15,
    }

    # ============================================================
    # Combine ID and isolation requirements
    # ============================================================

    # Iso MVA IDs already include isolation
    if electron_id_wp in ["wp80iso", "wp90iso"]:
        electron_id_iso_mask = id_wps[electron_id_wp]
    else:
        electron_id_iso_mask = (
            id_wps[electron_id_wp]
            & iso_wps[electron_iso_wp]
        )

    # ============================================================
    # Final electron selection mask
    # ============================================================

    return (
        electron_pt_mask
        & electron_eta_mask
        & electron_id_iso_mask
    )


