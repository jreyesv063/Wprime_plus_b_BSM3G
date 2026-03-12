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

    if not (dataset.startswith("TTTo") and lepton_flavor in ["tau" , "mu"]):
        return    

    if year not in {"2016APV", "2016", "2017", "2018"}:
        raise ValueError(f"Año no reconocido: {year}")

    if lepton_flavor not in {"tau", "mu"}:
        raise ValueError(f"Lepton flavor no reconocido: {lepton_flavor}")        

    cset = correctionlib.CorrectionSet.from_file(
        f"wprime_plus_b/corrections/top_boost/top_boost_{lepton_flavor}_{year}.json"
    )

    lepton_map = {"ele": objects["electrons"], "mu": objects["muons"], "tau": objects["taus"]}
    
    lepton = lepton_map[lepton_flavor]

    ST = ak.to_numpy(
            ak.sum(lepton.pt, axis=1) 
            + objects["met"].pt 
            + ak.sum(objects["bjets"].pt, axis=1) 
            + ak.sum(objects["lightjets"].pt, axis=1) 
            + ak.sum(objects["topjets"].pt, axis=1) 
            + ak.sum(objects["wjets"].pt, axis=1)
    )

    nj = ak.to_numpy(objects["events"].njets_noTopTagger)

    # Máscara con todas las condiciones
    selection_mask = (
        (ak.num(lepton) == 1) # Se tenga al menos un lepton
        & (objects["events"].top_tagger_case_id > 0)
    )

    if lepton_flavor  == "mu":
        # Evaluar peso y aplicar máscara
        sf = ak.where(
            selection_mask,
            cset["top_boost_weight"].evaluate(ST, nj, "nominal"),
            1.0
        )
    else:
        # Evaluar peso y aplicar máscara
        sf = ak.where(
            selection_mask,
            cset["top_boost_weight"].evaluate(ST, nj, "nominal"),
            1.0
        )        

    weights.add(
        name=f"top_boost_weight_{lepton_flavor}_{year}",
        weight=sf,
    )
