import json
import numpy as np
import awkward as ak
import correctionlib
from typing import Type
from typing import Tuple
from typing import Optional, Union
from coffea.analysis_tools import Weights
from wprime_plus_b.corrections.utils import get_pog_json


# ============================================================================
#   Met XY corrections
# ============================================================================
def apply_met_phi_corrections(
    events: ak.Array,
    year: str,
    syst_var: bool 
) -> Tuple[ak.Array, ak.Array]:
    """
    Apply MET φ modulation (XY) corrections to account for detector effects.
    
    Corrects MET for azimuthal modulation patterns caused by detector geometry, pileup, and other experimental effects. These are also called "XY corrections" because they correct both MET magnitude and direction in the transverse plane.
    
    Different implementations for Run2 (MET) vs Run3 (PuppiMET) with different systematic uncertainty treatments.
        
    Correction Types
    ----------------
    Run2 (MET φ Corrections):
    ------------------------
    Separate corrections for:
    - MC: correction_MC_pt, correction_MC_phi
    - Data: correction_Data_pt, correction_Data_phi
    
    Corrections depend on: MET_pt, MET_φ, npvsGood, run (data only)
    
    Run3 (PuppiMET XY Corrections):
    --------------------------------
    Single correction with variations:
    - pt, phi: Nominal corrections
    - pu_up, pu_dn: Pileup systematic variations
    
    Corrections depend on: PuppiMET_pt, PuppiMET_φ, npvsGood, era, sample type
    
    Notes
    -----
    - Run2 uses Type-1 corrected MET; Run3 uses PuppiMET
    - Run3 era mapping: "2022_pre"→"2022", "2022_post"→"2022EE", etc.
    - For data, run number is used; for MC, run=0 is passed
    - Should be applied after all other MET corrections (JES, JER, leptons)
    
    
    References
    ----------
    - CMS MET φ corrections: https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETRun2Corrections
    - Run3 PuppiMET: https://twiki.cern.ch/twiki/bin/view/CMS/MissingETRun3
    - POG JSON: https://gitlab.cern.ch/cms-nanoaod/jsonpog-integration
    
    """
    # ===========================================================
    #  Read json file: corrections
    # ===========================================================
    # Correction name
    with open("wprime_plus_b/corrections/correction_names/JME.json", "r") as f:
        names = json.load(f)["met_xyCorrections"][year]   
    

    # =============================================================
    # Correction: event-level weight (nominal/up/down)
    # =============================================================   
    cset = correctionlib.CorrectionSet.from_file(get_pog_json(json_name="met", year=year))
        
    met_type = "MET" if year in ["2016APV", "2016", "2017", "2018"] else "PuppiMET"
    run_number = events.run # 0.0 if hasattr(events, "genWeight") else events.run
    

    if met_type == "MET":
       

        correction_pt_name =  names["MC"]["pt"] if hasattr(events, "genWeight") else names["Data"]["pt"]       
        correction_phi_name = names["MC"]["phi"] if hasattr(events, "genWeight") else names["Data"]["phi"]

        # 1. Definimos la máscara para no exceder los límites de la corrección
        mask = (
            (events[met_type, "pt"] < 6499.0)
        )

        #  Evitar error con la muestra: root://xcache//store/data/Run2016F/MET/NANOAOD/HIPM_UL2016_MiniAODv2_NanoAODv9-v2/40000/8AC67EF9-0B54-6546-A0A6-9BECD936E67D.root
        if year == "2016APV":
            mask = mask & (events.run < 278771)
            
        
        # 2. Extraemos los datos protegidos por la máscara
        # Usamos fill_none para que correctionlib no falle con valores NaN
        met_pt = ak.fill_none(events[met_type, "pt"].mask[mask], 0.0)
        met_phi = ak.fill_none(events[met_type, "phi"].mask[mask], 0.0)
        npvs = ak.fill_none(events.PV.npvsGood.mask[mask], 0.0)
        runs = ak.fill_none(run_number.mask[mask], ak.min(run_number))
        

        # 3. Calculamos las correcciones usando los datos filtrados
        corrected_pt = cset[correction_pt_name].evaluate(
            met_pt, 
            met_phi, 
            npvs, 
            runs
        )
        corrected_phi = cset[correction_phi_name].evaluate(
            met_pt, 
            met_phi, 
            npvs, 
            runs
        )

        # 4. Actualizar el objeto original usando ak.where
        # Los eventos fuera de la máscara conservan su valor original (sin corrección)
        events[met_type, "pt"] = ak.where(mask, corrected_pt, events[met_type, "pt"])
        events[met_type, "phi"] = ak.where(mask, corrected_phi, events[met_type, "phi"])

  
        # ===========================================
        #  Systematic variations
        # ===========================================
        if syst_var and hasattr(events, "genWeight"):
            fields = events[met_type].fields
            up_fields = [f for f in fields if f.lower().endswith("up")]
            down_fields = [f for f in fields if f.lower().endswith("down")]

            pt_vals = {}
            phi_vals = {}
        
            for case in up_fields + down_fields:
                value = events[met_type, case]
            
                kind, rest = case.split("_", 1)  # kind = 'pt' o 'phi'
            
                if kind == "pt":
                    pt_vals[rest] = value
                elif kind == "phi":
                    phi_vals[rest] = value
            
            for corr_name in pt_vals.keys():
                # 1. Definimos la máscara específica para esta variación
                # Es importante usar el pt_val de la variación para la máscara
                mask_syst = (pt_vals[corr_name] < 6499.0)  

                # 2. Preparamos los inputs enmascarados
                syst_pt = ak.fill_none(pt_vals[corr_name].mask[mask_syst], 0.0)
                syst_phi = ak.fill_none(phi_vals[corr_name].mask[mask_syst], 0.0)
                syst_npvs = ak.fill_none(events.PV.npvsGood.mask[mask_syst], 0.0)
               
                # 3. Evaluamos la corrección nominal sobre los valores variados
                # CMS suele aplicar la corrección JERC nominal incluso sobre las variaciones
                corr_pt_val = cset[correction_pt_name].evaluate(
                    syst_pt, 
                    syst_phi, 
                    syst_npvs, 
                    run_number
                )
                corr_phi_val = cset[correction_phi_name].evaluate(
                    syst_pt, 
                    syst_phi, 
                    syst_npvs, 
                    run_number
                )
                

                # 4. Guardamos el resultado usando ak.where 
                # Si está fuera de rango, mantenemos el valor de la variación sin corregir
                events[met_type, f"pt_{corr_name}"] = ak.where(
                    mask_syst, corr_pt_val, pt_vals[corr_name]
                )
                events[met_type, f"phi_{corr_name}"] = ak.where(
                    mask_syst, corr_phi_val, phi_vals[corr_name]
                )
                
    else:
        correction_name = names["pt"] 
        
        era = {
            "2022_pre": "2022",
            "2022_post": "2022EE",
            "2023_pre": "2023",
            "2023_post": "2023BPix"
        }
                
        met_pt_corr = cset[correction_name].evaluate(
            "pt",                                             # phi, phi_stat_xdn, phi_stat_xup, phi_stat_ydn, phi_stat_yup, pt, pt_stat_xdn, pt_stat_xup, pt_stat_ydn, pt_stat_yup
            met_type,                                         # MET, PuppiMET
            era[year] ,                                       # 2022; 2022EE; 2023; 2023BPix
            "MC" if hasattr(events, "genWeight") else "DATA", # DATA, MC
            "nom",                                            # nom, pu_dn, pu_up  
            events[met_type, "pt"],                           # met_pt
            events[met_type, "phi"],                          # met_phi
            events.PV.npvsGood                                # Number of vertices
        )
        
        events[met_type, "pt"] = met_pt_corr

        
        # ===========================================
        #  Systematic variations
        # ===========================================
        if syst_var and hasattr(events, "genWeight"):
            # up
            met_pt_corr_up = cset[correction_name].evaluate(
                "pt",
                met_type, 
                era[year] , 
                "MC" if hasattr(events, "genWeight") else "DATA", 
                "pu_up", 
                events.PuppiMET.pt, 
                events.PuppiMET.phi, 
                events.PV.npvsGood
            )
            events[met_type, "MET_xy_pt_up"] = met_pt_corr_up
            
            # down
            met_pt_corr_down = cset[correction_name].evaluate(
                "pt",
                met_type, 
                era[year] , 
                "MC" if hasattr(events, "genWeight") else "DATA", 
                "pu_dn", 
                events.PuppiMET.pt, 
                events.PuppiMET.phi, 
                events.PV.npvsGood
            )
            events[met_type, "MET_xy_pt_down"] = met_pt_corr_down
                    


# ============================================================================
#   MET unclustered
# ============================================================================
def apply_met_unclustered(events: ak.Array, syst_var: bool = True):
    """
    Compute unclustered energy systematic variations for MET. 
    
    Estimates systematic uncertainty from energy not clustered into reconstructed objects (jets, leptons, photons). The variation is computed by shifting MET by the pre-computed Δx and Δy components stored in NanoAOD.
    
    The unclustered energy uncertainty represents:
    - Energy from soft particles below reconstruction thresholds
    - Detector noise and miscalibrations
    - Energy outside jet cones and lepton isolation cones
    
    Variation Calculation
    ---------------------
    MET_x^up   = MET_x_nominal + Δx
    MET_y^up   = MET_y_nominal + Δy
    MET_x^down = MET_x_nominal - Δx
    MET_y^down = MET_y_nominal - Δy
    
    where Δx = MetUnclustEnUpDeltaX, Δy = MetUnclustEnUpDeltaY
    
    Notes
    -----
    - Only applied to MC samples (Δx, Δy only available in MC NanoAOD)
    - Δx and Δy are pre-computed during NanoAOD production
    - The variation is symmetric: down = -up (opposite direction)
    - Should be applied after all other MET corrections (JES, JER, lepton)
    
    References
    ----------
    - Met factory: https://github.com/btovar/coffea/blob/d033466d91303f3881fe4a65cc995a59066984f8/src/coffea/jetmet_tools/CorrectedMETFactory.py
    - twiki: https://twiki.cern.ch/twiki/bin/view/CMS/MissingETUncertaintyPrescription ;  https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD#MET
    """
    
    # Systematic variation only in MC samples
    if not hasattr(events, "genWeight") or syst_var == False:
        return
        
    # ===================================================
    #  MET (nominal values) 
    # ===================================================
    # MET.pt should include the other corrections
    met_x = events.MET.pt * np.cos(events.MET.phi)
    met_y = events.MET.pt * np.sin(events.MET.phi)

    # ====================================================
    #  Estimate unclustered variations
    # ====================================================
    dx = events.MET.MetUnclustEnUpDeltaX
    dy = events.MET.MetUnclustEnUpDeltaY


    # ====================================================
    #  Calculate Up and Down variations
    # ====================================================
    # Up
    met_x_up = met_x + dx
    met_y_up = met_y + dy

    met_pt_up =  np.sqrt(met_x_up**2 + met_y_up**2) 
    met_phi_up = np.arctan2(met_y_up, met_x_up)

    events["MET", "MET_pt_UnclusteredEnergy_up"] =  met_pt_up
    events["MET", "MET_phi_UnclusteredEnergy_up"] =  met_phi_up


    # Down
    met_x_down = met_x - dx
    met_y_down = met_y - dy


    met_pt_down =  np.sqrt(met_x_down**2 + met_y_down**2) 
    met_phi_down = np.arctan2(met_y_down, met_x_down)


    events["MET", "MET_pt_UnclusteredEnergy_down"] =  met_pt_down
    events["MET", "MET_phi_UnclusteredEnergy_down"] =  met_phi_down

    
# ============================================================================
#   Met trigger
# ============================================================================
def add_met_trigger_corrections(
    mask_trigger,
    dataset,
    met: ak.Array,
    weights: Type[Weights],
    year: str,
    trigger_mask = None
) -> Tuple[ak.Array, ak.Array]:
    """
    Apply MET trigger scale factor corrections and store them in a Weights container.

    The MET trigger scale factors are evaluated as a function of recoil-corrected
    MET and applied only to events passing the trigger mask. Events outside the
    trigger region receive a unit weight.
    """

    # ------------------------------------------------------------------
    # Select events within the trigger acceptance and extract MET
    # ------------------------------------------------------------------
    # Apply trigger mask to MET collection
    in_limit_met = met.mask[mask_trigger]

    # Use recoil-corrected MET; fill missing values with a safe default
    met_pt = ak.fill_none(in_limit_met.pt_recoil, 10.0)

    # ------------------------------------------------------------------
    # Load MET trigger scale factors from correctionlib: local file
    # ------------------------------------------------------------------
    cset = correctionlib.CorrectionSet.from_file(f"wprime_plus_b/corrections/HLT/tau/met_trigger_{year}_UL.json")

    # ------------------------------------------------------------------
    # Initialize scale factor arrays with unity (no correction by default)
    # ------------------------------------------------------------------
    nominal_sf = np.ones_like(met_pt)
    up_sf = np.ones_like(met_pt)
    down_sf = np.ones_like(met_pt)

    # ------------------------------------------------------------------
    # Apply dataset-dependent MET trigger scale factors
    # ------------------------------------------------------------------
    if dataset.startswith("WJetsToLNu"):
        # MET trigger SFs for W+jets background
        correction_name = "UL-MET-Trigger-SF_WJ"

        nominal_sf, up_sf, down_sf = [
            np.where(mask_trigger, cset[correction_name].evaluate(met_pt, var), 1.0)
            for var in ("nominal", "up", "down")
        ]

    elif dataset.startswith("TTTo"):
        # MET trigger SFs for tt̄ background
        correction_name = "UL-MET-Trigger-SF_TT"

        nominal_sf, up_sf, down_sf = [
            np.where(mask_trigger, cset[correction_name].evaluate(met_pt, var), 1.0)
            for var in ("nominal", "up", "down")
        ]
    
    # ------------------------------------------------------------------
    # Store MET trigger scale factors in the weights container
    # ------------------------------------------------------------------
    nominal_sf, up_sf, down_sf = [
        ak.where(trigger_mask, sf, 1.0)
        for sf in (nominal_sf, up_sf, down_sf)
    ]


    weights.add(
        name=f"CMS_eff_MET_trigger_{year}",
        weight=nominal_sf,
        weightUp=up_sf,
        weightDown=down_sf,
    )

                
# ============================================================================
#   Corrections due to modification of an object's pt.
# ============================================================================   
def update_met(
    events: ak.Array,
    met_initial: str = "RawMET",
    met_final: str = "MET",
    pt_new_objects: Optional[ak.Array] = None,
    pt_old_objects: Optional[ak.Array] = None,
    phi_objects: Optional[ak.Array] = None,
    variation: Optional[str] = None,
    add_delta: bool = False
) -> None:
    """
    Update MET after modifying object momenta (e.g., lepton corrections).
    
    Propagates changes in object momenta (leptons, photons, etc.) to MET by subtracting the momentum differences from the initial MET vector. This implements the standard MET correction formula when object energies change.
    
    Formula:
        MET_x^final = MET_x^initial - Σ[(pT_new^i - pT_old^i) × cos(φ^i)]
        MET_y^final = MET_y^initial - Σ[(pT_new^i - pT_old^i) × sin(φ^i)]
    
    Parameters
    ----------
    - met_initial: Name of the initial MET collection in events. Common options:
        - "RawMET": Uncorrected MET
        - "MET": JEC-corrected MET  
        - "PuppiMET": PUPPI MET
        - "RawPuppiMET": Raw PUPPI MET
        Default is "RawMET".
    - met_final: Name of the final/updated MET collection in events. Can be the same as met_initial to overwrite, or different to create a new field.
        Default is "MET".
    
    Notes
    -----
    - The function assumes φ remains unchanged for the corrected objects.
    - The correction is vectorial: each object contributes (ΔpT × cosφ, ΔpT × sinφ).
    - For multiple object types (e.g., electrons + muons), call separately for each.
    - If met_final == met_initial, the original MET is overwritten.
    - variation allows the calculated value to be classified, which is very useful for systematic variations.
    Typical Use Cases
    -----------------
    1. Lepton energy scale corrections (e.g., muon pT corrections (rochester); tau pT corrections (TES); JEC and JER)
    
    References
    ----------
    - CMS Type-1 MET Documentation: https://cms-jerc.web.cern.ch/Type1MET/
    """

    suffix = f"_{variation}" if variation else ""

    # ==============================================
    #  Initial values of MET
    # ==============================================
    # Initial values
    met_x_initial = events[met_initial, "pt"] * np.cos(events[met_initial, "phi"])
    met_y_initial = events[met_initial, "pt"] * np.sin(events[met_initial, "phi"])

    # get x and y changes
    delta_px = ak.sum((pt_new_objects - pt_old_objects) * np.cos(phi_objects), axis=1)
    delta_py = ak.sum((pt_new_objects - pt_old_objects) * np.sin(phi_objects), axis=1)

    if add_delta:
        delta_px = -delta_px
        delta_py = -delta_py

    met_corr_x = met_x_initial - delta_px
    met_corr_y = met_y_initial - delta_py


    # update MET fields: MET (pT, phi) components
    events[met_final, f"pt{suffix}"] = np.sqrt(met_corr_x**2 + met_corr_y**2)
    events[met_final, f"phi{suffix}"] = np.arctan2(met_corr_y, met_corr_x)

    # =================================================
    # update systematic variations
    # =================================================
    # Remember that if it is Raw, it will be a type I correction.
    # If variation is None, it means that the function is being used for a nominal value.   
    # Following the order in test processor: first is AK8, second is Rochester.
    if not met_initial.startswith("Raw") and variation == None:   
        fields = events[met_initial].fields
        up_fields = [f for f in fields if f.lower().endswith("up")]
        down_fields = [f for f in fields if f.lower().endswith("down")]

        pt_vals = {}
        phi_vals = {}
        
        for case in up_fields + down_fields:
            value = events[met_initial, case]
        
            kind, rest = case.split("_", 1)  # kind = 'pt' o 'phi'
        
            if kind == "pt":
                pt_vals[rest] = value
            elif kind == "phi":
                phi_vals[rest] = value
        
        for corr_name in pt_vals.keys():
            if corr_name not in phi_vals:
                continue        

            if add_delta:
                delta_px = -delta_px
                delta_py = -delta_py
                                    
            met_x_tmp = pt_vals[corr_name] * np.cos(phi_vals[corr_name]) - delta_px
            met_y_tmp = pt_vals[corr_name] * np.sin(phi_vals[corr_name]) - delta_py 
        
            events['MET', f"pt_{corr_name}"] = np.sqrt(met_x_tmp**2 + met_y_tmp**2)
            events['MET', f"phi_{corr_name}"] = np.arctan2(met_y_tmp, met_x_tmp)

# ============================================================================
#   New met variables
# ============================================================================ 

def met_recoil(events: ak.Array, muons) -> None:
    """
    Recompute MET recoil using selected muons only.

    The recoil-corrected MET is obtained by propagating the transverse
    momentum of reconstructed muons to the original MET vector.
    The updated MET components are stored as new fields ``pt_recoil``
    and ``phi_recoil`` in the ``events.MET`` collection.


    """

    # Original MET components
    met_pt = events.MET.pt
    met_phi = events.MET.phi

    # Muon transverse momentum components
    muons_pt = muons.pt
    muons_phi = muons.phi

    # ------------------------------------------------------------------
    # Propagate muon momenta to MET in Cartesian (x, y) components
    # ------------------------------------------------------------------
    recoil_px = (
        met_pt * np.cos(met_phi)
        + ak.sum(muons_pt * np.cos(muons_phi), axis=-1)
    )

    recoil_py = (
        met_pt * np.sin(met_phi)
        + ak.sum(muons_pt * np.sin(muons_phi), axis=-1)
    )

    # ------------------------------------------------------------------
    # Convert recoil-corrected MET back to (pT, phi)
    # ------------------------------------------------------------------
    recoil_pt = np.sqrt(recoil_px**2 + recoil_py**2)
    recoil_phi = np.arctan2(recoil_py, recoil_px)

    # ------------------------------------------------------------------
    # Store recoil-corrected MET in the events record
    # ------------------------------------------------------------------
    events["MET", "pt_recoil"] = recoil_pt
    events["MET", "phi_recoil"] = recoil_phi

