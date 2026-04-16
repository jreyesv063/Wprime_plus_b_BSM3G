from wprime_plus_b.processors.utils.utils_topXfinder import get_topXfinder_masks


def select_top_tagger(
    objects,
    region_mask,
    cross_cleaning,
    criteria
):
    """
    Evaluates top quark reconstruction using the topXfinder tool and generates 
    a selection mask based on the success or failure of the tagger.

    This function identifies top candidates for different topologies (Resolved, 
    Partially Resolved, or Merged). It can be used for nominal signal selection 
    or to define background-rich control regions (e.g., W+jets) by inverting 
    the tagger requirement.

    Args:
    -----
    objects: dict
        Dictionary containing physics objects and the events array.
    region_mask: ak.Array (boolean)
        Mask defining the phase space where the top tagger should be evaluated.
    cross_cleaning: ak.Array (boolean)
        Mask to ensure leptons and jets are properly separated in DeltaR.
    criteria: dict
        Configuration dictionary containing:
        - 'cases': List of specific top topologies to consider (e.g., cases 1-13).
        - 'nworkers': Number of parallel workers for the calculation.
        - 'invert_top_tagger': Boolean. If True, selects events where the tagger 
          was evaluated but no top candidate was found (ID == 0).

    Returns:
    --------
    objects: dict
        The input objects dictionary updated with the 'top_tagger_case_id' branch.
    top_mask: ak.Array (boolean)
        Selection mask for the analysis. Note: events with ID == -1 (not evaluated) 
        are always excluded from this mask.
    """

    objects = get_topXfinder_masks(
        objects=objects,           
        region_mask = region_mask, 
        cross_cleaning =cross_cleaning,
        top_tagger_cases=criteria["cases"],
        nworkers=criteria["nworkers"]
    )        

    

    if criteria["invert_top_tagger"]:
        # Remember: 0 means that the event was evaluated, but no top was found; -1 means that the event was not evaluated in the top tagger.
        top_mask = objects["events"].top_tagger_case_id == 0
        
    else:
        # Remember: 0 means that the event was evaluated, but no top was found; -1 means that the event was not evaluated in the top tagger.
        top_mask = objects["events"].top_tagger_case_id > 0


    return objects, top_mask