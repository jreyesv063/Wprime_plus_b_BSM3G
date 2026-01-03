import numpy as np
from typing import Type
from coffea.analysis_tools import Weights


def add_genweight_weight(
    events, weights_container: Type[Weights]
):
    """
    Add generator-level event weights to a Weights container.

    This function computes a sign-based generator weight from the
    ``events.genWeight`` field and adds it to the provided
    ``Weights`` container under the name ``"genweight"``.
    Events with positive generator weights receive a weight of +1,
    while events with negative generator weights receive a weight of -1.

    Parameters
    ----------
    events : awkward.Array
        Events array containing a ``genWeight`` attribute.
    weights_container : coffea.analysis_tools.Weights
        Weights container to which the generator weight will be added.

    Returns
    -------
    None
        The function modifies the ``weights_container`` in place.
    """
    # add genweights
    genweight_values = lambda events: np.where(events.genWeight > 0, 1, -1)

    weights_container.add(
        "genweight",
        weight=genweight_values(events),
    )

   