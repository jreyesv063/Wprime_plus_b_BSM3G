import numpy as np
import correctionlib
import awkward as ak
from typing import Type
from typing import Tuple
from coffea.analysis_tools import Weights
from wprime_plus_b.corrections.utils import get_pog_json


def add_top_boost_corrections(
    objects: dict,
    lepton_flavor: str,
    dataset: str,
    weights: Type[Weights],
    year: str
) -> None:


    if year not in {"2016APV", "2016", "2017", "2018"}:
        raise ValueError(f"Unrecognized year: {year}")

    if lepton_flavor not in {"tau", "mu"}:
        raise ValueError(f"Unrecognized lepton flavor: {lepton_flavor}")  

    if not (dataset.startswith("TTTo") and lepton_flavor in ["tau" , "mu"]):
        weights.add(
            name=f"top_boost_weight_{lepton_flavor}_{year}",
            weight=np.ones_like(objects["met"].pt),
            weightUp=np.ones_like(objects["met"].pt),
            weightDown=np.ones_like(objects["met"].pt),
        )
        return    

    cset = correctionlib.CorrectionSet.from_file(
        f"wprime_plus_b/corrections/top_boost/top_boost_{lepton_flavor}_{year}.json"
    )

    lepton_map = {"ele": objects["electrons"], "mu": objects["muons"], "tau": objects["taus"]}
    
    lepton = ak.firsts(lepton_map[lepton_flavor])

    ST = ak.fill_none(objects["events"].top_tagger_pt  + objects["met"].pt  + lepton.pt, 0)
    nj = ak.fill_none(objects["events"].njets_noTopTagger, 0)


    # Mask
    selection_mask = (
        (ak.num(lepton_map[lepton_flavor]) == 1)    # Exactly one lepton
        & (objects["events"].top_tagger_case_id > 0)
    )


    sf = {
        var: ak.where(
            selection_mask,
            cset["top_boost_weight"].evaluate(ST, nj, var),
            1.0
        )
        for var in ["nominal", "up", "down"]
    }

    weights.add(
        name=f"top_boost_weight_{lepton_flavor}_{year}",
        weight=sf['nominal'],
        weightUp=sf['up'],
        weightDown=sf['down'],
    )
