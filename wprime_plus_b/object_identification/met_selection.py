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
    syst_var: bool = True
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
    met_type = "MET" if year in ["2016APV", "2016", "2017", "2018"] else "PuppiMET"

    met = events[met_type]

    # ===============================================
    #          Build the MET selection mask
    # ===============================================
    def mask(met_pt, met_max, met_min, invert):
        if met_max is None:
            good_met = (met_pt >= met_min)
        
        else:
            good_met = ((met_pt >= met_min) & (met_pt <= met_max))

        if invert_met_cut:
            good_met = ~good_met

        return good_met

    met_mask = mask(met.pt, met_max, met_min, invert_met_cut)
    

    masks =  {"nominal": met_mask}

    # ============================================================
    # Systematic variations: AK4 (JES, JER); AK8 (JES, JER)
    # Tau (TES); Muon (Rochester); Electron (run 3)
    # ============================================================  
    if syst_var and hasattr(events, "genWeight"):
        variations = {
            "AK4": {
                "JES": ("pt_AK4_JES_up", "pt_AK4_JES_down"),
                "JER": ("pt_AK4_JER_up", "pt_AK4_JER_down"),
            },
            "AK8": {
                "JES": ("pt_AK8_JES_up", "pt_AK8_JES_down"),
                "JER": ("pt_AK8_JER_up", "pt_AK8_JER_down"),
            },
            "lepton": {
                "TES": ("pt_Tau_TES_up", "pt_Tau_TES_down"),
                "Rochester": ("pt_Muon_Rochester_up", "pt_Muon_Rochester_down"),
                "SS": ("pt_Electron_SS_up", "pt_Electron_SS_down"),  
            },
            "met": {
                "Uncluster": ("MET_pt_UnclusteredEnergy_up", "MET_pt_UnclusteredEnergy_down"),
            }
        }

        for group, systs in variations.items():
            group_dict = {}

            for syst_name, attrs in systs.items():
                up_attr, down_attr = attrs

                # Check if the attributes exist in the MET collection before applying the mask
                if hasattr(met, up_attr) and hasattr(met, down_attr):
                    group_dict[syst_name] = {
                        "up": mask(getattr(met, up_attr), met_max, met_min, invert_met_cut),
                        "down": mask(getattr(met, down_attr), met_max, met_min, invert_met_cut),
                    }

            # Only add the group to the result if the systematic variations are present
            if group_dict:
                masks[group] = group_dict 

    return masks


def select_good_delta_phi_jet_met(
    events: ak.Array,
    jets: ak.Array,
    delta_phi_cut: float,
    invert_delta_phi_cut: bool = False,
    year: str = "2017",
    syst_var: bool = True
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
    def mask(jet_phi, met_phi, min_cut,  invert_cut):

        # Deltaphi value: https://github.com/scikit-hep/coffea/blob/1f69f3a373740b4f916139545cff7e2d87d08116/coffea/nanoevents/methods/vector.py#L67-L68
        delta_phi = (jet_phi - met_phi[..., None] + np.pi) % (2*np.pi) - np.pi
        
        # Event passes if all jets satisfy the condition
        good_dphi = ak.all(np.abs(delta_phi) >= min_cut, axis=-1) #ak.all(ak.unflatten(good_delta_phi, nj), axis=-1)
        
        if invert_cut:
            good_dphi = ~good_dphi

        return good_dphi

    delta_phi_mask = mask(jets.phi, met.phi, delta_phi_cut, invert_delta_phi_cut)

    masks = {"nominal": delta_phi_mask}
    # ============================================================
    # Systematic variations: AK4 (JES, JER); AK8 (JES, JER)
    # Tau (TES); Muon (Rochester); Electron (run 3)
    # ============================================================  
    if syst_var and hasattr(events, "genWeight"):
        variations = {
            "AK4": {
                "JES": ("phi_AK4_JES_up", "phi_AK4_JES_down"),
                "JER": ("phi_AK4_JER_up", "phi_AK4_JER_down"),
            },
            "AK8": {
                "JES": ("phi_AK8_JES_up", "phi_AK8_JES_down"),
                "JER": ("phi_AK8_JER_up", "phi_AK8_JER_down"),
            },
            "lepton": {
                "TES": ("phi_Tau_TES_up", "phi_Tau_TES_down"),
                "Rochester": ("phi_Muon_Rochester_up", "phi_Muon_Rochester_down"),
                "SS": ("phi_Electron_SS_up", "phi_Electron_SS_down"),
            },
            "met": {
                "Uncluster": ("MET_phi_UnclusteredEnergy_up", "MET_phi_UnclusteredEnergy_down"),
            }
        }

        for group, systs in variations.items():
            group_dict = {}
        
            for syst_name, attrs in systs.items():
                up_attr, down_attr = attrs

                # Check if the attributes exist in the MET collection before applying the mask
                if hasattr(met, up_attr) and hasattr(met, down_attr):
                    group_dict[syst_name] = {
                        "up": mask(
                            jets.phi,
                            getattr(met, up_attr),
                            delta_phi_cut,
                            invert_delta_phi_cut,
                        ),
                        "down": mask(
                            jets.phi,
                            getattr(met, down_attr),
                            delta_phi_cut,
                            invert_delta_phi_cut,
                        ),
                    }

            # Only add the group to the result if the systematic variations are present
            if group_dict:
                masks[group] = group_dict


    return masks
        