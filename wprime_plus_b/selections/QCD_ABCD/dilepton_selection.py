import json
import numpy as np
import awkward as ak
import importlib.resources


def select_good_dilepton(
    leptons: ak.Array,
    mass_min: float = 60.0,
    mass_max: float = 120.0,
    charge_selection: str = "LS"
) -> ak.highlevel.Array:
   

    # Mass
    leading_lepton = ak.pad_none(leptons, 2)[:, 0]
    subleading_lepton = ak.pad_none(leptons, 2)[:, 1]

    dilepton = leading_lepton + subleading_lepton 

    good_mass = (
        (dilepton.mass >=  mass_min)
        & (dilepton.mass <=  mass_max)
    )

    # Charge
    charge = {
        "OS": leading_lepton.charge * subleading_lepton.charge < 0,
        "LS": leading_lepton.charge * subleading_lepton.charge > 0
    }

    Ql_Ql = charge[charge_selection]

    return good_mass  & Ql_Ql
    
