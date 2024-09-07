import correctionlib
import numpy as np
import awkward as ak
from typing import Type
from .utils import unflat_sf
from coffea.analysis_tools import Weights
from wprime_plus_b.corrections.utils import get_pog_json


def add_pujetid_weight(
    jets: ak.Array,
    weights: Type[Weights],
    year: str = "2017",
    working_point: str = "T",
    variation: str = "nominal",
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
    puid_wps = {
        "L": 4,
        "M": 6,
        "T": 7,
    }
    # flat jets array since correction function works only on flat arrays
    j, n = ak.flatten(jets), ak.num(jets)

    # get 'in-limits' jets
    jet_pt_mask = (j.pt > 20.0) & (j.pt < 57.5)
    jet_eta_mask = np.abs(j.eta) < 5.
    jet_puid_mask = j.puId == puid_wps[working_point]
    genjet_match_mask = j.genJetIdx >= 0
    in_jet_mask = jet_pt_mask & jet_eta_mask & jet_puid_mask & genjet_match_mask
    in_jets = j.mask[in_jet_mask]

    # get jet transverse momentum and pseudorapidity (replace None values with some 'in-limit' value)
    jets_pt = ak.fill_none(in_jets.pt, 20.0)
    jets_eta = ak.fill_none(in_jets.eta, 0.0)

    # define correction set
    cset = correctionlib.CorrectionSet.from_file(
        get_pog_json("pujetid", year)
    )
    # get nominal scale factors
    # If jet in 'in-limits' jets, then take the computed SF, otherwise assign 1
    # Unflatten to original shape
    nominal_sf = unflat_sf(
        cset["PUJetID_eff"].evaluate(jets_eta, jets_pt, "nom", working_point),
        in_jet_mask,
        n,
    )
    if variation == "nominal":
        # get 'up' and 'down' variations
        up_sf = unflat_sf(
            cset["PUJetID_eff"].evaluate(jets_eta, jets_pt, "up", working_point),
            in_jet_mask,
            n,
        )
        down_sf = unflat_sf(
            cset["PUJetID_eff"].evaluate(jets_eta, jets_pt, "down", working_point),
            in_jet_mask,
            n,
        )
        # add nominal, up and down scale factors to weights container
        weights.add(
            name=f"pujetid_{working_point}",
            weight=nominal_sf,
            weightUp=up_sf,
            weightDown=down_sf,
        )
    else:
        # add nominal scale factors to weights container
        weights.add(name=f"pujetid_{working_point}", weight=nominal_sf)