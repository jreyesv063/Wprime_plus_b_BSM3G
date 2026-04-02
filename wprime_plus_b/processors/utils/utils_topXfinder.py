import numpy as np
import awkward as ak
from concurrent.futures import ThreadPoolExecutor, as_completed
from wprime_plus_b.processors.utils.topXfinder import topXfinder
from wprime_plus_b.processors.utils.analysis_utils import apply_selection, parallel_processing 


def get_topXfinder_masks(
    objects,
    region_mask,
    lepton_flavor,
    cross_cleaning,
    top_tagger_cases,    
    nworkers=8
):

    # =======================================================
    # Mask with initial cuts
    # =======================================================    
    #region_mask = selections.all(*initial_cuts)
    
    # =======================================================
    # Reduce the objects to be considered in the top tagger
    # =======================================================
    selected_objects = apply_selection(objects, region_mask)
    
    # =================================
    # Create a topXfinder instance
    # =================================
    topX = topXfinder(
        lepton_flavor=lepton_flavor,
        bjets=selected_objects["bjets"],
        jets=selected_objects["lightjets"],
        topjets=selected_objects["topjets"],
        wjets=selected_objects["wjets"],
        cc=cross_cleaning,
    )
        
    escenarios = {
        "case_1": topX.Scenario_1jet_unresolve,
        "case_2": topX.Scenario_2jets_unresolve,
        "case_3": topX.Scenario_2jets_partiallyresolve,
        "case_4": topX.Scenario_3jets_partiallyresolve,
        "case_5": topX.Scenario_3jets_resolve,
        "case_6": topX.Scenario_4jets_resolve,
        "case_7": topX.Scenario_Njets_resolve,
        "case_8": topX.Scenario_Nbjets_resolve,
        "case_9": topX.Scenario_1jet_unresolve_general,
        "case_10": topX.Scenario_2jets_unresolve_general,
        "case_11": topX.Scenario_2jets_partiallyresolve_general,
        "case_12": topX.Scenario_3jets_partiallyresolve_general,
        "case_13": topX.Scenario_3jets_resolve_general,
    }


    # =================================
    # Parallelization per case
    # =================================    
    top_masses = {}   # Dictionary to store tops for each scenario
    top_pts = {}
    masks = {}  # Dictionary to store boolean masks for each scenario

    # Use as many threads as there are CPU cores
    with ThreadPoolExecutor(max_workers=nworkers) as ex:
        # Submit each scenario to the thread pool
        # - key: case with True
        # - escenarios[key]: method
        futures = [
            ex.submit(parallel_processing, key, escenarios[key])
            for key, active in top_tagger_cases.items() if active and key in escenarios
        ]    
        
        # Collect results as they finish (order may not match the original list)
        for fut in as_completed(futures):
            key, t, p, m = fut.result()  # Get the returned tuple from run_case
            top_masses[key] = t            # Store the top mass array for this scenario
            top_pts[key] = p              # Store the top pt array for this scenario
            masks[key] = m               # Store the boolean mask for this scenario
    
    
    """
             Save top tagger cases
    Assign a numeric case ID to the SELECTED events
    - Convention (events.top_tagger_case_id):
        -1 : top tagger has not been evaluated
        = 0: top tagger evaluated, but without tops reconstructed
        >0 : top tagger evaluated, top found in case X
    - Convention top mass (events.top_tagger_mass):
        -1 : top tagger has not been evaluated
        = 0: top tagger evaluated, but without tops reconstructed
        >0 : top tagger evaluated, top found with a mass m   
    """
    # Initial values
    case_id = ak.zeros_like(selected_objects["events"].event_index)
    top_mass = ak.zeros_like(selected_objects["events"].event_index, dtype=float)
    top_pt = ak.zeros_like(selected_objects["events"].event_index, dtype=float)

    for case_name in masks.keys():
        # Extract numeric ID from "case_X": Case_1 -> 1; etc.
        cid = int(case_name.split("_")[1])

         # Assign case ID where this mask is True
        case_id = ak.where(masks[case_name], cid, case_id)
        top_mass = ak.where(top_masses[case_name] > 0, top_masses[case_name], top_mass)
        top_pt = ak.where(top_pts[case_name] > 0, top_pts[case_name], top_pt)

    
    # Store case ID/mass in selected events
    selected_objects["events"] = ak.with_field(selected_objects["events"], top_mass, "top_tagger_mass")
    selected_objects["events"] = ak.with_field(selected_objects["events"], top_pt, "top_tagger_pt")
    selected_objects["events"] = ak.with_field(selected_objects["events"], case_id, "top_tagger_case_id")

    
    # Build a lookup table: event_index → top_tagger_case_id
    case_lookup = dict(
        zip(
            ak.to_list(selected_objects["events"].event_index),
            ak.to_list(selected_objects["events"].top_tagger_case_id),
        )
    )
    
    mass_lookup = dict(
        zip(
            ak.to_list(selected_objects["events"].event_index),
            ak.to_list(selected_objects["events"].top_tagger_mass),
        )
    )

    pt_lookup = dict(
        zip(
            ak.to_list(selected_objects["events"].event_index),
            ak.to_list(selected_objects["events"].top_tagger_pt),
        )
    )    

    # Fill full event arrays
    full_case_id = ak.Array([
        case_lookup.get(evt_idx, -1)
        for evt_idx in ak.to_list(objects["events"].event_index)
    ])
    
    full_top_mass = ak.Array([
        mass_lookup.get(evt_idx, -1.0)
        for evt_idx in ak.to_list(objects["events"].event_index)
    ])

    full_top_pt = ak.Array([
        pt_lookup.get(evt_idx, -1.0)
        for evt_idx in ak.to_list(objects["events"].event_index)
    ])

    
    # Store in full events
    objects["events"] = ak.with_field(
        objects["events"],
        full_case_id,
        "top_tagger_case_id",
    )
    
    objects["events"] = ak.with_field(
        objects["events"],
        full_top_mass,
        "top_tagger_mass",
    )

    objects["events"] = ak.with_field(
        objects["events"],
        full_top_pt,
        "top_tagger_pt",
    )
    
    # ============================================================================================
    # Determine the number of jets that did not participate in the reconstruction of the top.
    # ============================================================================================
    njets = ak.num(objects["bjets"]) + ak.num(objects["lightjets"]) + ak.num(objects["topjets"]) + ak.num(objects["wjets"])
    case_id = objects["events"].top_tagger_case_id
    
    # Start from njets, but default everything to 0
    njets_notop = ak.full_like(njets, 0)
    
    # Masks per case group
    mask_1 = (case_id == 1) | (case_id == 2) | (case_id == 9) | (case_id == 10)               
    mask_2 = (case_id == 3) | (case_id == 4) | (case_id == 11) | (case_id == 12)
    mask_3 = (case_id == 5) | (case_id == 6) | (case_id == 7) | (case_id == 8) | (case_id == 13)
    
    njets_notop = ak.where(mask_1, njets - 1, njets_notop) # Unresolved
    njets_notop = ak.where(mask_2, njets - 2, njets_notop) # Partially resolved
    njets_notop = ak.where(mask_3, njets - 3, njets_notop) # Resolved

    objects["events"]["njets_noTopTagger"] = njets_notop 
    
    
    #return object with the top tagger implementation
    return objects
        
