import correctionlib
import numpy as np
import awkward as ak
from .utils import unflat_sf
from typing import Type
from typing import Tuple
from wprime_plus_b.corrections.utils import get_pog_json
from coffea.analysis_tools import Weights


def add_top_boost_corrections(
    jets: ak.Array,
    bjets: ak.Array,
    muons: ak.Array,
    electrons: ak.Array,
    taus: ak.Array,
    met: ak.Array,
    lepton_flavor: str,
    dataset: str,
    weights: Type[Weights],
    year: str,
    variation: str = "nominal",
) -> Tuple[ak.Array, ak.Array]:
    
    
    if dataset.startswith('TTTo'):
        # get top boost correction, using the ST variable
        cset = correctionlib.CorrectionSet.from_file(
            "wprime_plus_b/data/top_boost.json"
        )

        jet_pt = ak.sum(jets.pt, axis=1)
        bjet_pt = ak.sum(bjets.pt, axis=1)
        njet = ak.num(jets)
        electron_pt = ak.sum(electrons.pt, axis=1)
        muon_pt = ak.sum(muons.pt, axis=1)
        tau_pt = ak.sum(taus.pt, axis=1)

        lepton = {
            "ele": electron_pt,
            "mu": muon_pt,
            "tau": tau_pt
        }

        st = lepton[lepton_flavor] + jet_pt + bjet_pt + met.pt

        # ST range
        in_st_mask = (
            (st >= 200.0)
            & (st <= 9000.0)
        )
        
        
        st_masked = st.mask[in_st_mask]
        
        st_pt = ak.fill_none(st_masked, 250)
        

        sf = cset[f"Top_boost_weight_{year}_UL_{lepton_flavor}"].evaluate(njet, st_pt ,"nominal")


        nominal_sf = np.where(in_st_mask, sf, 1.0)
    

        if variation == "nominal":
            # get 'up' and 'down' scale factors
            sf_up = cset[f"Top_boost_weight_{year}_UL_{lepton_flavor}"].evaluate(njet, st_pt ,"up")
            up_sf = np.where(in_st_mask, sf_up, 1.0)
            
            sf_down =  cset[f"Top_boost_weight_{year}_UL_{lepton_flavor}"].evaluate(njet, st_pt ,"down")
            down_sf = np.where(in_st_mask, sf_down, 1.0)
                    
            # add scale factors to weights container
            weights.add(
                name=f"top_boost_weight_{year}",
                weight=nominal_sf,
                weightUp=up_sf,
                weightDown=down_sf,
            )

        else:
            weights.add(
                name=f"top_boost_weight_{year}",
                weight=nominal_sf,
            )

    else:
        return
    