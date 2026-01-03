import numpy as np
import correctionlib
import awkward as ak
from typing import Type
from typing import Tuple
from coffea.analysis_tools import Weights
from wprime_plus_b.corrections.utils import get_pog_json


def add_top_boost_corrections(
    jets: ak.Array,
    bjets: ak.Array,
    fatjets: ak.Array,
    wjets: ak.Array,
    muons: ak.Array,
    electrons: ak.Array,
    taus: ak.Array,
    met: ak.Array,
    lepton_flavor: str,
    dataset: str,
    weights: Type[Weights],
    year: str,
    variation: str = "nominal",
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

    lepton_map = {"ele": electrons, "mu": muons, "tau": taus}
    lepton = lepton_map[lepton_flavor]

    casos = {
        "2016APV": {
            "njets": (0, 10),
            "ST": (300, 2000)
        },
        "2016": {
            "njets": (0, 10),
            "ST": (300, 2000)
        },
        "2017": {
            "njets": (0, 10),
            "ST": (300, 2000)
        },
        "2018": {
            "njets": (0, 10),
            "ST": (300, 2000)   
        }
    }

    # Calcular cantidades físicas para todos los eventos
    lepton_pt = ak.fill_none(ak.firsts(lepton.pt), 0.0)   # Solo un tau
    met_pt = ak.fill_none(met.pt, 0.0)
    jet_pt_sum = ak.fill_none(ak.sum(jets.pt, axis=1), 0.0)
    bjet_pt_sum = ak.fill_none(ak.sum(bjets.pt, axis=1), 0.0)
    fatjet_pt_sum = ak.fill_none(ak.sum(fatjets.pt, axis=1), 0.0)
    wjet_pt_sum = ak.fill_none(ak.sum(wjets.pt, axis=1), 0.0)

    ST = lepton_pt + met_pt + jet_pt_sum + bjet_pt_sum + fatjet_pt_sum + wjet_pt_sum

    jet_count = ak.num(jets)
    bjet_count = ak.num(bjets)
    fatjet_count = ak.num(fatjets)
    wjet_count = ak.num(wjets)

    njets = jet_count + bjet_count + fatjet_count + wjet_count

    # Máscara con todas las condiciones
    selection_mask = (
        (ak.num(lepton) == 1) # Se tenga al menos un lepton
        & (njets > casos[year]["njets"][0]) # se tenga njets mayor a 0
        & (ST >= casos[year]["ST"][0]) # ST sea mayor al minimo observado en los resultados
    )

    if lepton_flavor  == "mu":
        # Evaluar peso y aplicar máscara
        sf = ak.where(
            selection_mask,
            cset["top_boost_weight"].evaluate(njets, ST, "nominal"),
            1.0
        )
    else:
        # Evaluar peso y aplicar máscara
        sf = ak.where(
            selection_mask,
            cset["top_boost_weight"].evaluate(ST, njets, "nominal"),
            1.0
        )        

    weights.add(
        name=f"top_boost_weight_{lepton_flavor}_{year}",
        weight=sf,
    )

    