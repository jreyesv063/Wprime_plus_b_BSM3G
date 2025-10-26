import json
import numpy as np
import awkward as ak
import importlib.resources

from wprime_plus_b.processors.utils.analysis_utils import delta_r


def select_good_Z(
    electrons: ak.Array,
    muons: ak.Array,
    taus: ak.Array,
    lepton_flavor: str,
    cross_cleaning: float = 0.4,
    year: str = "2017",
    charge_selection: str = "OS",
    Z_mass_min: float = 71.0,
    Z_mass_max: float = 111.0,
) -> ak.highlevel.Array:
   
    # Selec lepton
    lepton_selection = {
        "tau": taus,
        "mu": muons,
        "ele": electrons
    }

    lepton = lepton_selection[lepton_flavor]

    leading_lepton = ak.pad_none(lepton, 2)[:, 0]
    subleading_lepton = ak.pad_none(lepton, 2)[:, 1]


    # Charge selection
    charge = {
        "OS": leading_lepton.charge * subleading_lepton.charge < 0,
        "LS": leading_lepton.charge * subleading_lepton.charge > 0
    }

    Ql_Ql = charge[charge_selection]

    # Mass selection
    good_Z = (
        ((leading_lepton + subleading_lepton).mass >=  Z_mass_min)
        & ((leading_lepton + subleading_lepton).mass <=  Z_mass_max)
    )

    # Delta R selection
    cross_cleaning = delta_r(leading_lepton, subleading_lepton, threshold = cross_cleaning)


    return good_Z & Ql_Ql & cross_cleaning
