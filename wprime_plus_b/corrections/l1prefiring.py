import awkward as ak
from typing import Type
from coffea.analysis_tools import Weights


def add_l1prefiring_weight(
    events: ak.Array, 
    weights_container: Type[Weights], 
    year: str
):
    """

    Source 1: https://twiki.cern.ch/twiki/bin/view/CMS/L1PrefiringWeightRecipe
    Source 2: https://muon-wiki.docs.cern.ch/guidelines/corrections/?h=l1#l1-trigger-prefiring
    
    Add pileup scale factor

    Run 2: In 2016 and 2017, the gradual timing shift of ECAL was not properly propagated to L1 trigger primitives (TP) resulting in a significant fraction of high eta TP being mistakenly associated to the previous bunch crossing.

    Run 3: In Run 3, L1 trigger prefiring was significantly reduced and currently, the L1 DPG does not recommend any corrections for the residual efficiency loss.
    """
    # add L1prefiring weights
    if year in ("2016", "2016APV", "2017"):
        weights_container.add(
            f"CMS_l1_ecal_prefiring_{year}",
            weight=events.L1PreFiringWeight.Nom,
            weightUp=events.L1PreFiringWeight.Up,
            weightDown=events.L1PreFiringWeight.Dn,
        )
