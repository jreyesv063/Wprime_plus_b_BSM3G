import correctionlib
import numpy as np
import awkward as ak
from typing import Type
from coffea.analysis_tools import Weights
from wprime_plus_b.corrections.utils import get_pog_json


def jetvetomaps_mask(jets: ak.Array, year: str, mapname: str = "jetvetomap"):
    """
    These are the jet veto maps showing regions with an excess of jets (hot zones) and lack of jets
    (cold zones). Using the phi-symmetry of the CMS detector, these areas with detector and or
    calibration issues can be pinpointed.

    Non-zero value indicates that the region is vetoed
    """
    hname = {
        "2016APV": "Summer19UL16_V1",
        "2016": "Summer19UL16_V1",
        "2018": "Summer19UL18_V1",
    }
    cset = correctionlib.CorrectionSet.from_file(get_pog_json("jetvetomaps", year))

    j, n = ak.flatten(jets), ak.num(jets)
    jet_eta_mask = np.abs(j.eta) < 5.19
    jet_phi_mask = np.abs(j.phi) < 3.14

    in_jet_mask = jet_eta_mask & jet_phi_mask
    in_jets = j.mask[in_jet_mask]

    jets_eta = ak.fill_none(in_jets.eta, 0.0)
    jets_phi = ak.fill_none(in_jets.phi, 0.0)

    vetomaps = cset[hname[year]].evaluate(mapname, jets_eta, jets_phi)
    vetomaps = ak.unflatten(vetomaps, n) == 0
    return vetomaps