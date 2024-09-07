import correctionlib
import numpy as np
import awkward as ak
from .utils import unflat_sf
from typing import Type
from typing import Tuple
from wprime_plus_b.corrections.utils import get_pog_json
from coffea.analysis_tools import Weights


def add_tau_high_pt_corrections(
    taus: ak.Array,
    weights: Type[Weights],
    year: str,
    variation: str = "nominal",
) -> Tuple[ak.Array, ak.Array]:
    
    
    
    # get tau high pt correction
    cset = correctionlib.CorrectionSet.from_file(
        "wprime_plus_b/data/tau_hightPt.json"
    )

    tau, n = ak.flatten(taus), ak.num(taus)

    in_tau_mask = (tau.pt >= 140.0)
    
    tau_masked = tau.mask[in_tau_mask]

    tau_pt = ak.fill_none(tau_masked.pt, 140)
    
    nominal_sf = unflat_sf(
        cset[f"UL-Tau-HightPt-SF_{year}"].evaluate(tau_pt, "nominal"),
        in_tau_mask,
        n        
    )
  
    
    if variation == "nominal":
        # get 'up' and 'down' scale factors
        sf_up = unflat_sf(
            cset[f"UL-Tau-HightPt-SF_{year}"].evaluate(tau_pt, "up"),
            in_tau_mask,
            n,
        )
        
        sf_down =  unflat_sf(
            cset[f"UL-Tau-HightPt-SF_{year}"].evaluate(tau_pt, "down"),
            in_tau_mask,
            n,
        )
                   
        # add scale factors to weights container
        weights.add(
            name=f"tau_highPt_{year}",
            weight=nominal_sf,
            weightUp=sf_up,
            weightDown=sf_down,
        )
    else:
        weights.add(
            name=f"tau_highPt_{year}",
            weight=nominal_sf,
        )
