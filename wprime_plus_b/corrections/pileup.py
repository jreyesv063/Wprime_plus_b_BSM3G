import json
import correctionlib
import awkward as ak
from typing import Type
from coffea.analysis_tools import Weights
from wprime_plus_b.corrections.utils import get_pog_json


def add_pileup_weight(
    events,
    weights_container: Type[Weights],
    year: str
) -> None:
    """
    add pileup scale factor

    Parameters:
    -----------
        events:
            Events array
        weights_container:
            Weight object from coffea.analysis_tools
        year:
            dataset year {'2016', '2017', '2018'}
        variation:
            if 'nominal' (default) add 'nominal', 'up' and 'down'
            variations to weights container. else, add only 'nominal' weights.
            
    https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/summaries/LUM_2017_UL_puWeights.html

    Source Run 2: https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun2
    Source Run 3: https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun3
    
    """
    # ===========================================================
    #  Read json file: corrections
    # ============================================================
    # Correction name
    with open("wprime_plus_b/corrections/correction_names/LUM.json", "r") as f:
        case = json.load(f)
    
    correction_name = case["CMS_pileup"][year]

    # =============================================================
    #  Number of true interactions
    # =============================================================   
    mask_nTrueInt = events.Pileup.nTrueInt < 100
    in_nti = events.Pileup.nTrueInt.mask[mask_nTrueInt]
    nti = ak.fill_none(in_nti, 1)

    # =============================================================
    # Correction: event-level weight (nominal/up/down)
    # =============================================================
    cset = correctionlib.CorrectionSet.from_file(get_pog_json(json_name="pileup", year=year))
    
    nominal_sf, up_sf, down_sf = [
        cset[correction_name].evaluate(ak.to_numpy(nti), v) for v in ("nominal", "up", "down")
    ]
    # add pileup scale factors to weights container
    weights_container.add(
        name=f"CMS_pileup_{year}",
        weight=nominal_sf,
        weightUp=up_sf,
        weightDown=down_sf,
    )