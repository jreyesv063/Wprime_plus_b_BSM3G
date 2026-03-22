import warnings
import numpy as np
import awkward as ak
import correctionlib
from wprime_plus_b.corrections.met import update_met
from wprime_plus_b.corrections.utils import get_pog_json
from wprime_plus_b.corrections.utils import sample_crystal_ball
from coffea.lookup_tools import txt_converters, rochester_lookup


# Run 2: https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2018
# Run 3: https://twiki.cern.ch/twiki/bin/view/CMS/MuonRun32022
# General: https://twiki.cern.ch/twiki/bin/viewauth/CMS/MuonPOG#User_Recommendations
# https://dasanalysissystem.docs.cern.ch/md__builds_cms-analysis_general_DasAnalysisSystem_Core_Installer_tables_Rochester_README.html
# Files: https://gitlab.cern.ch/akhukhun/roccor
# Run2 code taken from: https://github.com/deoache/bsm3g_coffea/blob/main/analysis/corrections/rochester.py
# Main reference: https://arxiv.org/pdf/1208.3710

def apply_rochester_corrections(
    events: ak.Array, 
    year: str = "2017",
    syst_var: bool = False
):
    """
    Apply Rochester muon momentum corrections for Run2 or Run3 data.
    
    This is a wrapper function that dispatches to the appropriate correction method based on the data-taking year. Rochester corrections differ between Run2 (2016-2018) and Run3 (2022+) due to methodology recommended.
    

    Modifies events in-place by calling either:
    - apply_rochester_corrections_run2() for Run2 years
    - apply_rochester_corrections_run3() for Run3 years
        
    Differences Between Run2 and Run3
    ---------------------------------
    - File Format:
       - Run2: Uses .txt files with custom format
       - Run3: Uses .json.gz files with correctionlib format
    
    Notes
    -----
    - The function automatically detects the era based on the year parameter
    - Both methods update MET after applying muon corrections
    """    
    if year in ["2016APV", "2016", "2017", "2018"]:
        apply_rochester_corrections_run2(events, year, syst_var)

    else:
        apply_rochester_corrections_run3(events, year, syst_var)
    


def apply_rochester_corrections_run2(
    events: ak.Array, 
    year: str = "2017",
    syst_var: bool = False
):
    """
    Apply Rochester muon momentum corrections and propagate to MET.
    
    Implements the Rochester muon momentum correction method for Monte Carlo and data. The corrections account for:
    - Momentum scale miscalibrations
    - Energy loss in detector material
    - Alignment effects
    
    For MC: Hybrid method using kSpread (matched) and kSmear (unmatched); For Data: Direct scaling with kScale. Also updates MET by propagating muon momentum changes.

    -------------------------
    MC Samples (has genWeight):
    --------------------------
    1. For muons with generator match (|ΔR| < 0.2): kSpreadMC(charge, pt, η, φ, genPt). Use a deterministic correction.     
    2. For muons without generator match: kSmearMC(charge, pt, η, φ, nTrackerLayers, rand). Use a tochastic smearing based on tracker layers and random number
    
    Data Samples:
    -------------
        correction = kScaleDT(charge, pt, η, φ). direct scale factor from data-driven calibration, similar to kSpreadMC.
    
    Notes
    -----
    - The Rochester corrections are multiplicative: pT_corr = pT × k
    - Systematic uncertainties are treated as absolute: pT_up = pT × (k + σ)
    - Random numbers for kSmear are generated per-muon for reproducibility
    - Run2 uses .txt files, Run3+ uses .json.gz files
    References
    ----------
    - Das: https://dasanalysissystem.docs.cern.ch/md__builds_cms-analysis_general_DasAnalysisSystem_Core_Installer_tables_Rochester_README.html
    - tool: https://github.com/annawoodard/coffea/blob/a3a9ae92e14e48f2c9b5362095e5868139d859ed/coffea/lookup_tools/rochester_lookup.py
    - twiki: https://twiki.cern.ch/twiki/bin/view/CMS/RochcorMuon
        
    See Also
    --------
    update_met : Function that updates MET after object momentum changes
    
    """

    # =============================================================
    #  Save initial conditions
    # =============================================================  
    events["Muon", "pt_nano"] = events.Muon.pt
    
    # ===========================================================
    #  Read local file
    # ===========================================================
    # https://gitlab.cern.ch/akhukhun/roccor 
    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/RochcorMuon
    rochester_data = txt_converters.convert_rochester_file(
        f"wprime_plus_b/corrections/Rochester/RoccoR{year}UL.txt", loaduncs=True
    )        

    rochester = rochester_lookup.rochester_lookup(rochester_data)
    
    # Ignore overflow warning
    warnings.filterwarnings("ignore", message="overflow encountered in power")
    

    if hasattr(events, "genWeight"):
        # Identifying whether information is available at the generator level
        # True is a match was detected, False otherwise.
        match = ~np.isnan(ak.fill_none(events.Muon.matched_gen.pt, np.nan))      #          
        hasgen_flat = np.array(ak.flatten(match))                                #

        # =======================================================
        # Initialize variables
        # =======================================================
        corrections = np.array(ak.flatten(ak.ones_like(events.Muon.pt)))
        errors = np.array(ak.flatten(ak.ones_like(events.Muon.pt)))
        
        # ========================================================
        # There is match, apply spread.
        # ========================================================
        mc_kspread = rochester.kSpreadMC(
            events.Muon.charge[match],
            events.Muon.pt[match],
            events.Muon.eta[match],
            events.Muon.phi[match],
            events.Muon.matched_gen.pt[match],
        )

        errspread = rochester.kSpreadMCerror(
            events.Muon.charge[match],
            events.Muon.pt[match],
            events.Muon.eta[match],
            events.Muon.phi[match],
            events.Muon.matched_gen.pt[match],
        )
        
        corrections[hasgen_flat] = np.array(ak.flatten(mc_kspread))
        errors[hasgen_flat] = np.array(ak.flatten(errspread))
        
        # ========================================================
        # If there is no match, apply smear.
        # Uniform random number [0, 1]
        # ========================================================
        mc_rand = np.random.rand(*ak.to_numpy(ak.flatten(events.Muon.pt)).shape)
        mc_rand = ak.unflatten(mc_rand, ak.num(events.Muon.pt, axis=1))
        
        mc_ksmear = rochester.kSmearMC(
            events.Muon.charge[~match],
            events.Muon.pt[~match],
            events.Muon.eta[~match],
            events.Muon.phi[~match],
            events.Muon.nTrackerLayers[~match],  
            mc_rand[~match],
        )    

        errsmear = rochester.kSmearMCerror(
            events.Muon.charge[~match],
            events.Muon.pt[~match],
            events.Muon.eta[~match],
            events.Muon.phi[~match],
            events.Muon.nTrackerLayers[~match],
            mc_rand[~match],
        )

        corrections[~hasgen_flat] = np.array(ak.flatten(mc_ksmear))
        errors[~hasgen_flat] = np.array(ak.flatten(errsmear))


        # ========================================================
        #  Results for montecarlo
        # ========================================================
        corrections = ak.unflatten(corrections, ak.num(events.Muon.pt, axis=1))
        errors = ak.unflatten(errors, ak.num(events.Muon.pt, axis=1))

    else:
        corrections = rochester.kScaleDT(
            events.Muon.charge, 
            events.Muon.pt, 
            events.Muon.eta, 
            events.Muon.phi
        )
        
        errors = rochester.kScaleDTerror(
            events.Muon.charge, 
            events.Muon.pt, 
            events.Muon.eta, 
            events.Muon.phi
        )     

    good_muons = (
        (events.Muon.pt_nano >= 30)
        & (events.Muon.pt_nano <= 200)
        & (events.Muon.nTrackerLayers > 0)
    )

    # Compute and save pt in Muons
    events["Muon", "pt"] = ak.where(good_muons, events.Muon.pt_nano * corrections, events.Muon.pt_nano)
    
    update_met(
        events = events,
        met_initial = "MET",
        met_final = "MET",
        pt_new_objects =  events.Muon.pt,
        pt_old_objects = events.Muon.pt_nano,
        phi_objects = events.Muon.phi,
        add_delta = True
    )
    # ====================================================
    # Systematic variations
    # ====================================================
    # Systematic variation and MC samples
    if syst_var and hasattr(events, "genWeight"):
        # Up
        events["Muon", "pt_up"] = ak.where(good_muons, events.Muon.pt_nano * (corrections + errors), events.Muon.pt_nano)
        update_met(
            events = events,
            met_initial = "MET",
            met_final = "MET",
            pt_new_objects =  events.Muon.pt_up,
            pt_old_objects = events.Muon.pt_nano,
            phi_objects = events.Muon.phi,
            variation = "Muon_Rochester_up",
            add_delta = True
        )

        # Down
        events["Muon", "pt_down"] = ak.where(good_muons, events.Muon.pt_nano * (corrections - errors), events.Muon.pt_nano)
        update_met(
            events = events,
            met_initial = "MET",
            met_final = "MET",
            pt_new_objects =  events.Muon.pt_down,
            pt_old_objects = events.Muon.pt_nano,
            phi_objects = events.Muon.phi,
            variation = "Muon_Rochester_down",
            add_delta = True
        )


# =============================================================================
#            Run 3
# =============================================================================
def mask_corrections(muons):

    muon_mask = (
        (muons.pt > 26.0)
        & (muons.pt < 200.0)
    )

    return muon_mask
    
def apply_rochester_corrections_run3(
    events: ak.Array, 
    is_mc: bool = False,
    year: str = "2024",
    variation: bool = False
):        

    """
    Ref: https://github.com/Vvvvvvvictor/HiggsZGammaAna/blob/010741748567a1986a12948c41bbed70d146c61a/HiggsDNA/higgs_dna/systematics/lepton_systematics.py#L1030-L1281
    In contrast to Run 2, for Run 3 we have centralized correction. All parameters are available, and calculations must be performed manually.

    - Data: only scale correction
    - MC: scale + smear correction, with up/down variations
    
    """

    # ===========================================================
    #  Read json file: corrections
    # ============================================================
    # Correction name
    with open("wprime_plus_b/corrections/correction_names/MUO.json", "r") as f:
        case = json.load(f)["MediumPt"]["CMS_scale_m"][year]

    events["Muon", "pt_raw"] = events.Muon.pt

    # Get nominal, up, and down scale factors
    cset = correctionlib.CorrectionSet.from_file(get_pog_json(json_name="muon", year=year))
        
    # =============================================================
    # Select muon candidates
    # =============================================================     
    muons_flatten, n_muons = ak.flatten(events.Muon), ak.num(events.Muon)
    mask = mask_corrections(muons_flatten)
    muons_filter = muons_flatten.mask[mask]

    muon_pt = ak.fill_none(muons_filter.pt, 30.0)
    muon_eta = ak.fill_none(muons_filter.eta, 0.0)
    muon_phi = ak.fill_none(muons_filter.phi, 0.0)
    muon_charge = ak.fill_none(muons_filter.charge, 1.0)
    muon_nTrackerLayers =  ak.fill_none(muons_filter.nTrackerLayers, 10.0) 

    # =============================================================
    # Load parameters
    # =============================================================  
    if is_mc:
        # =========================================
        #  MC
        # =========================================
        # ---------------------------------
        # Step 1: Scale
        # ---------------------------------        
        A_mc =  cset[case["A_MC"]].evaluate(muon_eta, muon_phi, "nom") 
        M_mc =  cset[case["M_MC"]].evaluate(muon_eta, muon_phi, "nom") 

        # Calculate k factor:  k_{scale} =  1/[M + Q * A * pT^{raw}]
        k_scale = 1.0 / (M_mc  + muon_charge * A_mc * muon_pt)

        #  pT^{corr} = pT^{raw} * K_{scale}.  Remove arbitrary values filled with fill_none to avoid unnecessary values.
        pt_scale_tmp = ak.where(mask, muon_pt * k_scale, muons_flatten.pt)


        # ---------------------------------
        # Step 2: Smear (Resolution)
        # ---------------------------------
        # Crystall ball
        cb_mean = cset[case["crystall_ball"]].evaluate(np.abs(muon_eta), muon_nTrackerLayers, 0)
        cb_sigma = cset[case["crystall_ball"]].evaluate(np.abs(muon_eta), muon_nTrackerLayers, 1)
        cb_n = cset[case["crystall_ball"]].evaluate(np.abs(muon_eta), muon_nTrackerLayers, 2)
        cb_alpha = cset[case["crystall_ball"]].evaluate(np.abs(muon_eta), muon_nTrackerLayers, 3)

        invcdf = sample_crystal_ball(cb_mean, cb_sigma, cb_alpha, cb_n, len(muon_pt))

        # Polinomial
        poly_p0 = cset[case["sigma"]].evaluate(np.abs(muon_eta), muon_nTrackerLayers, 0)
        poly_p1 = cset[case["sigma"]].evaluate(np.abs(muon_eta), muon_nTrackerLayers, 1)
        poly_p2 = cset[case["sigma"]].evaluate(np.abs(muon_eta), muon_nTrackerLayers, 2)

        #  sigma = p0 + p1 * pT^{scale} + p2 * pT^{scale} * pT^{scale} 
        sigma = poly_p0 + poly_p1 * pt_scale_tmp + poly_p2 * pt_scale_tmp * pt_scale_tmp
        sigma = np.maximum(sigma, 0.0)
        
        
        # Get k factor: k_{Data} and  k_{MC}. Consider only scenarios k_{Data} >  k_{MC}
        k_data = cset[case["k_Data"]].evaluate(np.abs(muon_eta), "nom")
        k_mc = cset[case["k_MC"]].evaluate(np.abs(muon_eta), "nom")       

        k_factor = np.sqrt(np.maximum(k_data**2 - k_mc**2, 0.0))

        # x variable
        x = k_factor * sigma * invcdf

        # k_{spread} = 1 / [1 + x]
        k_spread = 1 / (1 + x)

        # pT^{corr} =  k_{scale} * k_{spread} * pT^{raw} =  k_{spread} * pT^{scale} 
        # Remove arbitrary values filled with fill_none to avoid unnecessary values.
        pt_corr_tmp = ak.where(mask, k_spread * pt_scale_tmp, muons_flatten.pt)


        # Save new muon pt
        events["Muon", "pt"] = ak.unflatten(pt_corr_tmp, n_muons)    

        # ---------------------------------
        # Step 3: Scale Up/Down variations
        #---------------------------------
        if variation:
            stat_a = cset[case["A_MC"]].evaluate(muon_eta, muon_phi, "stat")
            stat_m = cset[case["M_MC"]].evaluate(muon_eta, muon_phi, "stat")
            stat_rho = cset[case["M_MC"]].evaluate(muon_eta, muon_phi, "rho_stat")
    
            # Calculating uncertainty
            scale_unc = pt_corr_tmp * np.sqrt( (stat_m/pt_corr_tmp)**2 + stat_a**2 + 2 * muon_charge * stat_rho * stat_m/pt_corr_tmp * stat_a )
            scale_unc = ak.where(mask, scale_unc, 0.0)

            muon_list = {
                "nom": events.Muon.pt,
                "up": events.Muon.pt + scale_unc,
                "down": events.Muon.pt - scale_unc,
                "phi": events.Muon.phi
            }
        
    
    else:
        # =========================================
        #  Data
        # =========================================
        # ---------------------------------
        # Step 1: Scale
        # ---------------------------------          
        A_data =  cset[case["A_data"]].evaluate(muon_eta, muon_phi, "nom") 
        M_data =  cset[case["M_data"]].evaluate(muon_eta, muon_phi, "nom") 

        # Calculate k factor:  k_{scale} =  1/[M + Q * A * pT^{raw}]
        k_scale = 1.0 / (M_data  + muon_charge * A_data * muon_pt)
        
        #  pT^{corr} = pT^{raw} * K_{scale}.  Remove arbitrary values filled with fill_none to avoid unnecessary values.
        pt_scale_tmp = ak.where(mask, muon_pt * k_scale, muons_flatten.pt)


        # Save new muon pt
        events["Muon", "pt"] = ak.unflatten(pt_scale_tmp, n_muons)

