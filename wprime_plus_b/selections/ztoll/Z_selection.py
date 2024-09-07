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
    

    # open and load btagDeepFlavB working point
    with importlib.resources.open_text("wprime_plus_b.data", "btagWPs.json") as file:
        btag_threshold = json.load(file)["deepJet"][year][btag_working_point]

    # break up selection for low and high pT jets
    low_pt_jets_mask = (
        (jets.pt > jet_pt_threshold)
        & (jets.pt < 50)
        & (np.abs(jets.eta) < jet_eta_threshold)
        & (jets.jetId == jet_id)
        & (jets.puId == puid_wps[jet_pileup_id])
        & (jets.btagDeepFlavB < btag_threshold )
    )

    high_pt_jets_mask = (
        (jets.pt >= 50)
        & (np.abs(jets.eta) < jet_eta_threshold)
        & (jets.jetId == jet_id)
        & (jets.btagDeepFlavB < btag_threshold )
    )

    return ak.where(
        (jets.pt > jet_pt_threshold) & (jets.pt < 50),
        low_pt_jets_mask,
        high_pt_jets_mask,
    )





