import json
import numpy as np
import awkward as ak
import importlib.resources


def select_good_Z(
    Z: ak.Array,
    leading_lepton: ak.Array,
    subleading_lepton: ak.Array,
    year: str = "2017",
    charge_selection: str = "OS",
    Z_mass_min: float = 80.0,
    Z_mass_max: float = 100.0,
) -> ak.highlevel.Array:
   
    charge = {
        "OS": leading_lepton.charge * subleading_lepton.charge < 0,
        "LS": leading_lepton.charge * subleading_lepton.charge > 0
    }

    Ql_Ql = charge[charge_selection]

    good_Z = (
        (Z.mass >=  Z_mass_min)
        & (Z.mass <=  Z_mass_max)
    )

    return good_Z & Ql_Ql
    