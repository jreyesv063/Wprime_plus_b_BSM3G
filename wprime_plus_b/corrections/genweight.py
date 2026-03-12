import numpy as np
import awkward as ak
from typing import Type
from coffea.analysis_tools import Weights


def add_genweight_weight(
    events: ak.Array, 
    weights_container: Type[Weights]
):
    """

    Source: https://answers.launchpad.net/mg5amcnlo/+question/269191  
    Source: https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD
    
    Add generator-level event weights to a Weights container.

    
    This function computes a sign-based generator weight from the events.genWeigh field and adds it to the provided  Weights container.
    
    - Events with positive generator weights receive a weight of +1,
    - Events with negative generator weights receive a weight of -1.

    """
    # add genweights
    genweight_values = lambda events: np.where(events.genWeight > 0, 1, -1)

    weights_container.add(
        "genweight",
        weight=genweight_values(events),
    )