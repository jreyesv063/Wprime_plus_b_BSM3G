import json
import copy
import numpy as np
import awkward as ak
import correctionlib
from wprime_plus_b.corrections.met import update_met
from wprime_plus_b.corrections.utils import get_pog_json


# ----------------------------------------------------------------------------------- #
# -- The tau energy scale (TES) corrections for taus are provided  ------------------ #
# --  to be applied to reconstructed tau_h Lorentz vector ----------------------------#
# --  It should be applied to a genuine tau -> genmatch = 5 --------------------------#
# -----------  (pT, mass and energy) in simulated data -------------------------------#
# tau_E  *= tes
# tau_pt *= tes
# tau_m  *= tes
# https://github.com/cms-tau-pog/TauIDsfs/tree/master
# ----------------------------------------------------------------------------------- #


def mask_corrections(tau):
    # https://github.com/cms-tau-pog/TauFW/blob/4056e9dec257b9f68d1a729c00aecc8e3e6bf97d/PicoProducer/python/analysis/ETauFakeRate/ModuleETau.py#L320
    # https://gitlab.cern.ch/cms-tau-pog/jsonpog-integration/-/blob/TauPOG_v2/POG/TAU/scripts/tau_tes.py
    """
    Create selection mask for tau leptons eligible for energy scale corrections.
    
    Applies standard CMS tau identification and selection criteria to determine which tau candidates should receive Tau Energy Scale (TES) corrections.
    The mask selects taus that are:
    1. Genuine or misidentified leptons (gen-matched)
    2. With valid decay modes (1 or 3 prongs)
    3. Within detector acceptance (|η| < 2.5)
    
    Notes
    -----
    - The mask is used before applying Tau Energy Scale (TES) corrections
    - Only taus passing this mask should receive TES corrections
    
    
    See Also
    --------
    apply_tau_corrections : Function that applies TES corrections using this mask
    """
    # ============================================================
    # Load tau information
    # ============================================================
    with open("wprime_plus_b/json_files/tau.json", "r") as f:
        taus_info = json.load(f)


    # ============================================================
    # Tau requirements
    # ============================================================    
    tau_dm_mask = ak.zeros_like(tau.decayMode, dtype=bool)
    for decay_mode in taus_info["prongs"]["1or3prongs"]:
        tau_dm_mask = tau_dm_mask | (tau.decayMode == decay_mode)

    tau_genmatch_mask = (
        (tau.genPartFlav == taus_info["genPartFlav"]["prompt_electron"])
        | (tau.genPartFlav == taus_info["genPartFlav"]["prompt_muon"])
        | (tau.genPartFlav == taus_info["genPartFlav"]["hadronic_tau_decay"])
        | (tau.genPartFlav == taus_info["genPartFlav"]["unmatched"])
    )
    tau_eta_mask = (np.abs(tau.eta) < 2.5)

    tau_mask = tau_genmatch_mask & tau_dm_mask  & tau_eta_mask
    
    return tau_mask


def apply_tau_energy_scale_corrections(
    events: ak.Array,
    year: str = "2017",
    syst_var: bool = False,
):
    """   
    The correction scales all tau four-momentum components uniformly:
        E_corr = E × tes
        pT_corr = pT × tes  
        m_corr = m × tes

    Workflow
    --------
    1. Save original tau kinematics as *_nano fields
    2. Load TES correction names from JSON configuration
    3. Select taus using mask_corrections()
    4. Load correctionlib TES corrections for given year
    5. Evaluate TES factors (nominal, up, down)
    6. Apply uniform scaling to pT, mass, and energy
    7. Update tau fields with corrected values
    8. Update MET using update_met()
    9. Compute systematic variations if requested
    
    Notes
    -----
    - TES is only applied to MC samples (requires generator truth)
    - The scaling is uniform: all four-momentum components scaled equally

    References
    ----------
    - Repository: https://github.com/cms-tau-pog/TauIDsfs/tree/master
    - Twiki: https://twiki.cern.ch/twiki/bin/view/CMS/TauIDRecommendation    
    """
    # TES is only applied to MC
    if not hasattr(events, "genWeight"):
        return
    
    # ===========================================================
    #  Read json file: corrections
    # ============================================================
    # Correction name
    with open("wprime_plus_b/corrections/correction_names/TAU.json", "r") as f:
        case = json.load(f)

    correction_name = case["CMS_scale_t_DeepTau"][year]
    

    # =============================================================
    #  Save initial conditions
    # =============================================================  
    events["Tau", "pt_nano"] = events.Tau.pt
    events["Tau", "mass_nano"] = events.Tau.mass
    events["Tau", "E_nano"] = events.Tau.E

    
    # =============================================================
    # Select tau candidates
    # =============================================================      
    # Flatten taus and apply mask
    taus_flatten, ntaus = ak.flatten(events.Tau), ak.num(events.Tau)
    mask = mask_corrections(taus_flatten)
    taus_filter = taus_flatten.mask[mask]

    # Fill None values and get scale factors
    tau_pt, tau_eta, tau_dm, tau_genmatch = (ak.fill_none(taus_filter[field], 0) for field in ["pt", "eta", "decayMode", "genPartFlav"])
    tau_id_algorithm = "DeepTau2017v2p1" if year in ["2016APV", "2016", "2017", "2018"] else "DeepTau2018v2p5"
    
    
    # =============================================================
    # Correction: event-level weight (nominal/up/down)
    # =============================================================        
    # Get nominal, up, and down scale factors
    cset = correctionlib.CorrectionSet.from_file(get_pog_json(json_name="tau", year=year))

    scale_factors  = {var:
        cset[correction_name].evaluate(tau_pt, tau_eta, tau_dm, tau_genmatch, tau_id_algorithm, var) 
        for var in ("nom", "up", "down")
    }

    # =============================================================
    #  Calculating changes in kinematic variables: pt, mass
    # ============================================================= 
    # Compute new pt and mass values. taus_new is a map: nom; up; down.
    taus_new = {var: (taus_filter.pt_nano * scale_factors[var], taus_filter.mass_nano * scale_factors[var], taus_filter.E_nano * scale_factors[var]) for var in scale_factors}
    
    # Create corrected arrays with the same size as taus_flatten. Eliminate arbitrary numbers filled with "fill_none".
    taus_corrected = {
        var: [
            ak.where(mask, taus_new[var][0], taus_flatten.pt),    # Corrected pt
            ak.where(mask, taus_new[var][1], taus_flatten.mass),  # Corrected mass
            ak.where(mask, taus_new[var][2], taus_flatten.E)      # Corrected energy
        ]
        for var in scale_factors
    } # [0] -> pt; [1] -> mass; [2] -> Energy

    # Unflatten and update events
    events["Tau", "pt"] = ak.unflatten(taus_corrected["nom"][0], ntaus)
    events["Tau", "mass"] = ak.unflatten(taus_corrected["nom"][1], ntaus)


    update_met(
        events = events,
        met_initial = "MET",
        met_final = "MET",
        pt_new_objects =  events.Tau.pt,
        pt_old_objects = events.Tau.pt_nano,
        phi_objects = events.Tau.phi,
        add_delta = True
    )

    # ====================================================
    # Systematic variations
    # ====================================================
    # Systematic variation and MC samples
    if syst_var and hasattr(events, "genWeight"):
        # Up
        events["Tau", "pt_up"] = ak.unflatten(taus_corrected["up"][0], ntaus)
        update_met(
            events = events,
            met_initial = "MET",
            met_final = "MET",
            pt_new_objects =  events.Tau.pt_up,
            pt_old_objects = events.Tau.pt_nano,
            phi_objects = events.Tau.phi,
            variation = "Tau_TES_up",
            add_delta = True
        )

        # Down
        events["Tau", "pt_down"] = ak.unflatten(taus_corrected["down"][0], ntaus)
        update_met(
            events = events,
            met_initial = "MET",
            met_final = "MET",
            pt_new_objects =  events.Tau.pt_down,
            pt_old_objects = events.Tau.pt_nano,
            phi_objects = events.Tau.phi,
            variation = "Tau_TES_down",
            add_delta = True
        )
    

    
