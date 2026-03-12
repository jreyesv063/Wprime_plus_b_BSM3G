import numpy as np
import awkward as ak


def get_stitching_mask(
    events: ak.Array,
    dataset_name: str,
) -> ak.Array:
    """
    Apply generator-level HT stitching to inclusive DY and WJets samples.

    This function removes overlap between inclusive and HT-binned samples
    by keeping only events with low generator HT in the inclusive datasets.

    The HT selection is applied only to specific datasets and ignored
    otherwise.

    Parameters
    ----------
    events : ak.Array
        NanoEvents object containing generator-level information
        (must include `events.LHE.HT`).

    dataset_name : str
        Name of the dataset being processed.

    Returns
    -------
    ak.Array
        Boolean event mask. True = event is kept.
    """

    # ------------------------------------------------------------
    # Datasets that require HT stitching
    # ------------------------------------------------------------
    ht_filtered_datasets = [
        "DYJetsToLL_M-50_inclusive",
        "DYJetsToLL_M-50_ext",
        "WJetsToLNu_inclusive",
        "WJetsToLNu_ext",
    ]

    # ------------------------------------------------------------
    # Default: keep all events
    # ------------------------------------------------------------
    stitching_mask = np.ones(len(events), dtype=bool)

    # ------------------------------------------------------------
    # Apply HT cut only to inclusive samples
    # ------------------------------------------------------------
    if any(
        dataset_name.startswith(pattern) and "_HT-" not in dataset_name
        for pattern in ht_filtered_datasets
    ):
        lower_gen_ht = 0.0
        upper_gen_ht = 70.0

        stitching_mask = (
            (events.LHE.HT >= lower_gen_ht)
            & (events.LHE.HT < upper_gen_ht)
        )

    return stitching_mask