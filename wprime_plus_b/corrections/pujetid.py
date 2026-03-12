import json
import numpy as np
import awkward as ak
import correctionlib
from typing import Type
from .utils import unflat_sf
from coffea.analysis_tools import Weights
from wprime_plus_b.corrections.utils import get_pog_json


def add_pujetid_weight(
    jets: ak.Array,
    weights: Type[Weights],
    year: str = "2017",
    working_point: str = "Tight",
    jet_mask = None
):
    """
    add jet pileup ID scale factor

    Parameters:
    -----------
        jets:
            Jet collection
        weights:
            Weights object from coffea.analysis_tools
        year:
            dataset year {'2016', '2017', '2018'}
        working_point:
            pujetId working point {'L', 'M', 'T'}
        variation:
            if 'nominal' (default) add 'nominal', 'up' and 'down'
            variations to weights container. else, add only 'nominal' weights.
    """

    # ===========================================================
    #  Read json files
    # ============================================================
    # Correction name
    with open("wprime_plus_b/corrections/correction_names/JME.json", "r") as f:
        case = json.load(f)

    correction_name = case["CMS_eff_j_PUJetID"][year]   
    
    with open("wprime_plus_b/json_files/jet.json", "r") as f:
        puid_wps = json.load(f)["pujetid"][year]

    # =============================================================
    #  Jet candidates
    # =============================================================  
    # flat jets array since correction function works only on flat arrays
    j, n = ak.flatten(jets), ak.num(jets)

    # get 'in-limits' jets
    jet_pt_mask = (j.pt <= 50.0)
    jet_eta_mask = (np.abs(j.eta) <= 5.0)
    jet_puid_mask = (j.puId == puid_wps[working_point])
    genjet_match_mask = (j.genJetIdx >= 0)
    in_jet_mask = jet_pt_mask & jet_eta_mask & jet_puid_mask & genjet_match_mask
    in_jets = j.mask[in_jet_mask]

    # get jet transverse momentum and pseudorapidity (replace None values with some 'in-limit' value)
    jets_pt = ak.fill_none(in_jets.pt, 20.0)
    jets_eta = ak.fill_none(in_jets.eta, 0.0)



    # =============================================================
    # Correction: event-level weight (nominal/up/down)
    # =============================================================    
    # define correction set
    cset = correctionlib.CorrectionSet.from_file(get_pog_json("pujetid", year))

    
    # Get nominal, up, and down scale factors
    nominal_sf, up_sf, down_sf = [
        unflat_sf(cset[correction_name].evaluate(jets_eta, jets_pt, v, working_point[0]), in_jet_mask, n)
        for v in ("nom", "up", "down")
    ]

    nominal_sf, up_sf, down_sf = [
        ak.where(jet_mask, sf, 1.0)
        for sf in (nominal_sf, up_sf, down_sf)
    ]
    
    # add nominal, up and down scale factors to weights container
    weights.add(
        name=f"CMS_eff_j_PUJetID_eff_{year}",
        weight=nominal_sf,
        weightUp=up_sf,
        weightDown=down_sf,
    )