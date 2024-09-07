import correctionlib
import awkward as ak
from typing import Type
from coffea.analysis_tools import Weights
from wprime_plus_b.corrections.utils import get_pog_json


def add_pileup_weight(
    events,
    weights_container: Type[Weights],
    year: str,
    variation: str = "nominal",
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
    """
    # define correction set and goldenJSON file names
    cset = correctionlib.CorrectionSet.from_file(
        get_pog_json(json_name="pileup", year=year)
    )
    year_to_corr = {
        "2016APV": "Collisions16_UltraLegacy_goldenJSON",
        "2016": "Collisions16_UltraLegacy_goldenJSON",
        "2017": "Collisions17_UltraLegacy_goldenJSON",
        "2018": "Collisions18_UltraLegacy_goldenJSON",
    }
    # get number of true interactions
    mask_nTrueInt = events.Pileup.nTrueInt < 100
    in_nti = events.Pileup.nTrueInt.mask[mask_nTrueInt]
    nti = ak.fill_none(in_nti, 1)

    # get nominal scale factors
    nominal_sf = cset[year_to_corr[year]].evaluate(ak.to_numpy(nti), "nominal")
    if variation == "nominal":
        # get up and down variations
        up_sf = cset[year_to_corr[year]].evaluate(ak.to_numpy(nti), "up")
        down_sf = cset[year_to_corr[year]].evaluate(ak.to_numpy(nti), "down")
        # add pileup scale factors to weights container
        weights_container.add(
            name="pileup",
            weight=nominal_sf,
            weightUp=up_sf,
            weightDown=down_sf,
        )
    else:
        weights_container.add(
            name="pileup",
            weight=nominal_sf,
        )