import correctionlib
import numpy as np
import awkward as ak
from .utils import unflat_sf
from typing import Type
from typing import Tuple
from wprime_plus_b.corrections.utils import get_pog_json
from coffea.analysis_tools import Weights


def add_ttbar_boost_corrections(
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
            f"wprime_plus_b/data/ttbar_boost_{lepton_flavor}.json"
        )

        jet_pt = ak.sum(jets.pt, axis=1)
        bjet_pt = ak.sum(bjets.pt, axis=1)
        njets = ak.num(jets) + ak.num(bjets)
        electron_pt = ak.sum(electrons.pt, axis=1)
        muon_pt = ak.sum(muons.pt, axis=1)
        tau_pt = ak.sum(taus.pt, axis=1)

        lepton = {
            "ele": electron_pt,
            "mu": muon_pt,
            "tau": tau_pt
        }

        st = lepton[lepton_flavor] + jet_pt + bjet_pt + met.pt
        

        sf_nominal = cset[f"ttbar_boost_weight_{year}_UL"].evaluate(njets, st ,"nominal")

   

        if variation == "nominal":
            # get 'up' and 'down' scale factors
            sf_up = cset[f"ttbar_boost_weight_{year}_UL"].evaluate(njets, st ,"up")
            sf_down =  cset[f"ttbar_boost_weight_{year}_UL"].evaluate(njets, st ,"down")
                    
            # add scale factors to weights container
            weights.add(
                name=f"ttbar_boost_weight_{year}_{lepton_flavor}",
                weight=sf_nominal,
                weightUp=sf_up,
                weightDown=sf_down,
            )

        else:
            weights.add(
                name=f"top_boost_weight_{year}_{lepton_flavor}",
                weight=sf_nominal,
            )

    else:
        return
    
    