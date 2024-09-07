from typing import Type
from coffea.analysis_tools import Weights


def add_l1prefiring_weight(
    events, weights_container: Type[Weights], year: str, variation: str = "nominal"
):
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
    """
    # add L1prefiring weights
    if year in ("2016", "2016APV", "2017"):
        if variation == "nominal":
            weights_container.add(
                "L1Prefiring",
                weight=events.L1PreFiringWeight.Nom,
                weightUp=events.L1PreFiringWeight.Up,
                weightDown=events.L1PreFiringWeight.Dn,
            )
        else:
            weights_container.add(
                "L1Prefiring",
                weight=events.L1PreFiringWeight.Nom,
            )