import awkward as ak
from wprime_plus_b.processors.utils.topXfinder import topXfinder
from wprime_plus_b.processors.utils.analysis_utils import apply_selection, top_tagger, njets_no_used_top_tagger


def get_topXfinder_masks(
    lepton_flavor,
    region_mask,
    objects,
    top_tagger_cases,
    cross_cleaning,
    invert_topXfinder=False,
):
    """
    Run the topX finder and top tagger, and compute masks and jet counters
    after removing objects used in the top reconstruction.

    Parameters
    ----------
    lepton_flavor : str
        Lepton channel (e.g. 'electron', 'muon').
    region_mask : ak.Array
        Event-level mask defining the analysis region.
    objects : dict
        Dictionary containing physics objects (jets, bjets, fatjets, wjets, etc.).
    top_tagger_cases : dict
        Dictionary enabling/disabling individual top-tagger cases.
    cross_cleaning : object
        Cross-cleaning configuration passed to the topX finder.

    Returns
    -------
    mask_top : ak.Array
        Boolean mask of events passing the top tagger.
    njets_no_top : dict
        Dictionary with jet multiplicities after removing top-tagged objects.
    tops : ak.Array
        Array indicating reconstructed top candidates.
    """

    # Collect only the enabled top-tagger cases
    cases = [
        case for case, enabled in top_tagger_cases.items()
        if enabled
    ]

    # Apply the region selection to all physics objects
    selected_objects = apply_selection(objects, region_mask)

    # Run the topX finder to build top candidates
    topX = topXfinder(
        lepton_flavor=lepton_flavor,
        bjets=selected_objects["bjets"],
        jets=selected_objects["jets"],
        fatjets=selected_objects["fatjets"],
        wjets=selected_objects["wjets"],
        cc=cross_cleaning,
    )

    # Apply the top tagger and retrieve:
    #  - tops: reconstructed top candidates
    #  - mask_top: event-level mask for top-tagged events
    #  - masks: per-case masks indicating which objects are used
    tops, mask_top, masks = top_tagger(topX, top_tagger_cases=cases)

    # Definition of object multiplicities used by each top-tagger case
    escenarios = {
        **{f"case_{i}": {"njets": 0, "nbjets": 0, "nfatjets": 1, "nwjets": 0} for i in (1, 2, 9, 10)},
        **{f"case_{i}": {"njets": 0, "nbjets": 1, "nfatjets": 0, "nwjets": 1} for i in (3, 4, 11, 12)},
        **{f"case_{i}": {"njets": 2, "nbjets": 1, "nfatjets": 0, "nwjets": 0} for i in (5, 6, 7, 8, 13)},
    }

    # Use one of the masks as a template to initialize counters
    template_mask = next(iter(masks.values()))

    # Initialize jet multiplicities after removing top-tagged objects
    njets_no_top = {
        key: ak.zeros_like(template_mask, dtype=int)
        for key in (
            "njets_no_top",
            "nbjets_no_top",
            "nfatjets_no_top",
            "nwjets_no_top",
        )
    }

    # Loop over all top-tagger cases and subtract used objects
    for case_name, mask_case in masks.items():
        njets_no_top = njets_no_used_top_tagger(
            jets=selected_objects["jets"],
            bjets=selected_objects["bjets"],
            fatjets=selected_objects["fatjets"],
            wjets=selected_objects["wjets"],
            **escenarios[case_name],
            prev_counts=njets_no_top,
            mask=mask_case,
        )

    if invert_topXfinder:
        mask_top = ~mask_top

    return mask_top, masks, njets_no_top, tops, selected_objects
