import json
import numpy as np
import awkward as ak
from pathlib import Path
import importlib.resources
from typing import Optional
from coffea.nanoevents.methods.base import NanoEventsArray


def select_good_met(
    events: ak.Array,
    met_min: float,
    met_max: Optional[float] = None,
    invert_met_cut: bool = False,
    year: str = "2017",
) -> ak.highlevel.Array:
    """
    Selects events based on missing transverse energy (MET) requirements
    and returns a boolean mask.

    Parameters
    ----------
    events : ak.Array
        NanoEvents array containing event-level information.

    met_min : float
        Minimum MET threshold.

    met_max : float or None, optional
        Maximum MET threshold. If None, no upper bound on MET is applied.

    invert_met_cut : bool, optional
        If True, the MET selection is inverted.

    year : str
        Data-taking year. Determines whether MET or PuppiMET is used.

    Returns
    -------
    ak.Array
        Boolean mask selecting events that satisfy the MET criteria.
    """

    # Select the appropriate MET collection depending on the data-taking year
    # Run 2 uses MET and Run 3 uses Puppi MET
    if year in ["2016APV", "2016", "2017", "2018"]:
        met = events.MET
    elif year in ["2022_pre", "2022_post", "2023_pre", "2023_post", "2024"]:
        met = events.PuppiMET
    else:
        # Protect against unsupported or misspelled year values
        raise ValueError(f"Year {year} not recognized for MET selection.")

    # ===============================================
    #          Build the MET selection mask
    # ===============================================

    # If met_max is not provided, apply only a lower MET cut
    if met_max is None:
        good_met = met.pt > met_min
    else:
        good_met = ((met.pt > met_min) 
                    & (met.pt < met_max)
        )

    # Optionally invert the MET selection
    if invert_met_cut:
        return ~good_met
    else:
        return good_met



def select_good_delta_phi_jet_met(
    events: ak.Array,
    jets: ak.Array,
    delta_phi_cut: float,
    invert_delta_phi_cut: bool = False,
    year: str = "2017",
) -> ak.highlevel.Array:
    # --------------------------
    # TLorentz vectors: https://coffea-hep.readthedocs.io/en/v0.7.23/api/coffea.nanoevents.methods.vector.LorentzVector.html
    # Vectors: https://coffea-hep.readthedocs.io/en/v0.7.23/api/coffea.nanoevents.methods.vector.TwoVector.html#coffea.nanoevents.methods.vector.TwoVector.delta_phi
    # delta_phi is defined between [-pi, pi) ->  (a - b + numpy.pi) % (2 * numpy.pi) - numpy.pi
    # --------------------------     

    # Select the appropriate MET collection depending on the data-taking year
    # Run 2 uses MET and Run 3 uses Puppi MET
    if year in ["2016", "2017", "2018"]:
        met = events.MET
    elif year in ["2022_pre", "2022_post", "2023_pre", "2023_post", "2024"]:
        met = events.PuppiMET
    else:
        # Protect against unsupported or misspelled year values
        raise ValueError(f"Year {year} not recognized for MET selection.")


    # ===============================================
    #       Build the deltaphi selection mask
    # ===============================================

    # Compute delta-phi between each jet and MET
    delta_phi_met_jet = jets.delta_phi(met)

    # Event passes if ALL jets satisfy the delta-phi requirement
    good_delta_phi = ak.all(
        np.abs(delta_phi_met_jet) >= delta_phi_cut,
        axis=-1,
    )
    # Optionally invert the delta-phi selection
    if invert_delta_phi_cut:
        return ~good_delta_phi
    
    else:
        return good_delta_phi
