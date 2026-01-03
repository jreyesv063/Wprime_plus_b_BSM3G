import correctionlib
import numpy as np
import awkward as ak
from typing import Type
from typing import Tuple
from wprime_plus_b.corrections.utils import get_pog_json
from coffea.analysis_tools import Weights


def apply_met_phi_corrections(
    events: ak.Array,
    is_mc: bool,
    year: str,
) -> Tuple[ak.Array, ak.Array]:
    """
    Apply MET phi modulation corrections

    Parameters:
    -----------
        events:
            Events array
        is_mc:
            True if dataset is MC
        year:
            Year of the dataset {'2016', '2016APV', '2017', '2018'}

    Returns:
    --------
        corrected MET pt and phi
    """
    cset = correctionlib.CorrectionSet.from_file(
        get_pog_json(json_name="met", year=year)
    )
    events["MET", "pt_raw"] = ak.ones_like(events.MET.pt) * events.MET.pt
    events["MET", "phi_raw"] = ak.ones_like(events.MET.phi) * events.MET.phi
    
    # make sure to not cross the maximum allowed value for uncorrected met
    met_pt = events.MET.pt_raw
    met_pt = np.clip(met_pt, 0.0, 6499.0)
    met_phi = events.MET.phi_raw
    met_phi = np.clip(met_phi, -3.5, 3.5)

    # use correct run ranges when working with data, otherwise use uniform run numbers in an arbitrary large window
    run_ranges = {
        "2016APV": [272007, 278771],
        "2016": [278769, 284045],
        "2017": [297020, 306463],
        "2018": [315252, 325274],
    }
    data_kind = "mc" if is_mc else "data"
    if data_kind == "mc":
        run = np.random.randint(
            run_ranges[year][0], run_ranges[year][1], size=len(met_pt)
        )
    else:
        run = events.run
    try:
        events["MET", "pt"] = cset[f"pt_metphicorr_pfmet_{data_kind}"].evaluate(
            met_pt.to_numpy(), met_phi.to_numpy(), events.PV.npvsGood.to_numpy(), run
        )
        events["MET", "phi"] = cset[f"phi_metphicorr_pfmet_{data_kind}"].evaluate(
            met_pt.to_numpy(), met_phi.to_numpy(), events.PV.npvsGood.to_numpy(), run
        )
    except:
        pass



def update_met_list(events: ak.Array, syst_name: str, syst_var: bool, object: list) -> None:
    """
    helper function to compute new MET after lepton pT correction. 
    It uses the 'pt_raw' and 'pt' fields from 'leptons' to update MET 'pt' and 'phi' fields
    
    Parameters:
        - events:
            Events array
        - lepton:
            Lepton name {'Muon', 'Tau'}

    https://github.com/columnflow/columnflow/blob/16d35bb2f25f62f9110a8f1089e8dc5c62b29825/columnflow/calibration/util.py#L42
    https://github.com/Katsch21/hh2bbtautau/blob/e268752454a0ce0089ff08cc6c373a353be77679/hbt/calibration/tau.py#L117
    """
    assert syst_name in ["Muon", "Electron", "Tau", "FatJet_JES", "FatJet_JER", "Jet"], "Object not provided"


    # # Object without corrections
    # if object != "FatJet":
    #     # get needed lepton and MET fields
    #     object_pt_raw = events[syst_name, "pt_raw"]
    # else: 
    #     object_pt_raw = events[syst_name, "pt_raw_original"]   

    if syst_name == "FatJet_JES" or syst_name == "FatJet_JER":

        object_pt_raw = object["raw"]
    else:
        object_pt_raw = events[syst_name].pt_raw        

    # if hasattr(events[syst_name], "pt_raw_original" ):
    #     #object_pt_raw = events[syst_name].pt_raw_original
    #     object_pt_raw = object["raw"]
    # else:
    #     object_pt_raw = events[syst_name].pt_raw

    object_pt = {
            "up": object["up"], #up_object.pt,
            "nom": object["nom"], #nom_object.pt,
            "down": object["down"], # down_object.pt
    }


    object_phi = object["phi"]


    # MET with nominal values
    met_pt = events.MET.pt
    met_phi = events.MET.phi

    # build px and py sums before and after: we sum the time at x and the time at y of each event 
    if syst_name == "FatJet_JES" or syst_name == "FatJet_JER":   
        old_px =  ak.sum(object_pt_raw * np.cos(object_phi), axis=-1)
        old_py = ak.sum(object_pt_raw * np.sin(object_phi), axis=-1)

    else:
        old_px = ak.sum(object_pt_raw * np.cos(object_phi), axis=-1)
        old_py = ak.sum(object_pt_raw * np.sin(object_phi), axis=-1)

    old_px = ak.sum(object_pt_raw * np.cos(object_phi), axis=-1)
    old_py = ak.sum(object_pt_raw * np.sin(object_phi), axis=-1)

    new_px = {
        "up": ak.sum(object_pt["nom"] * np.cos(object_phi), axis=-1),
        "nom": ak.sum(object_pt["nom"] *  np.cos(object_phi), axis=-1),
        "down": ak.sum(object_pt["nom"] *  np.cos(object_phi), axis=-1)
    } 
    
    new_py = {
        "up": ak.sum(object_pt["nom"] * np.sin(object_phi), axis=-1),
        "nom": ak.sum(object_pt["nom"] *  np.sin(object_phi), axis=-1),
        "down": ak.sum(object_pt["down"] *  np.sin(object_phi), axis=-1)
    } 
   
    # get x and y changes
    delta_x = {
        "up": new_px["up"] - old_px,
        "nom": new_px["nom"] - old_px,
        "down": new_px["down"] - old_px
    }

    delta_y = {
        "up": new_py["up"] - old_py,
        "nom": new_py["nom"] - old_py,
        "down": new_py["down"] - old_py
    }



    # propagate changes to MET (x, y) components: Negative signs have been changed
    met_px = {
        "up": met_pt * np.cos(met_phi) + delta_x["up"],
        "nom": met_pt * np.cos(met_phi) + delta_x["nom"],
        "down": met_pt * np.cos(met_phi) + delta_x["down"]
    }
    met_py = {
        "up": met_pt * np.sin(met_phi) + delta_y["up"],
        "nom": met_pt * np.sin(met_phi) + delta_y["nom"],
        "down": met_pt * np.sin(met_phi) + delta_y["down"]
    }

    
    # propagate changes to MET (pT, phi) components
    met_pt = {
        f"{syst_name}":
            {
                "up": np.sqrt((met_px["up"] ** 2.0 + met_py["up"] ** 2.0)),
                "nom": np.sqrt((met_px["nom"] ** 2.0 + met_py["nom"] ** 2.0)),
                "down": np.sqrt((met_px["down"] ** 2.0 + met_py["down"] ** 2.0))
            }
    }

    met_phi = {
        f"{syst_name}":
            {
                "up": np.arctan2(met_py["up"], met_px["up"]),
                "nom": np.arctan2(met_py["nom"], met_px["nom"]),
                "down": np.arctan2(met_py["down"], met_px["down"])
            }
    }

    # Save the delta X and delta Y variations
    delta_var_list = {
        f"{syst_name}": 
            {
                "delta_x": {
                    "nom":  delta_x["nom"],
                    "up": delta_x["up"],
                    "down": delta_x["down"]
                },
                
                "delta_y": {
                    "nom": delta_y["nom"],
                    "up": delta_y["up"],
                    "down": delta_y["down"]
                }
            }
    }    


    # Overwrite the MET fields with the nominal values
    events["MET", "pt"] = met_pt[f"{syst_name}"]["nom"]
    events["MET", "phi"] = met_phi[f"{syst_name}"]["nom"]

    return met_pt, met_phi, delta_var_list


    

def update_met(events: ak.Array, lepton: str = "Muon") -> None:
    """
    helper function to compute new MET after lepton pT correction. 
    It uses the 'pt_raw' and 'pt' fields from 'leptons' to update MET 'pt' and 'phi' fields
    
    Parameters:
        - events:
            Events array
        - lepton:
            Lepton name {'Muon', 'Tau'}

    https://github.com/columnflow/columnflow/blob/16d35bb2f25f62f9110a8f1089e8dc5c62b29825/columnflow/calibration/util.py#L42
    https://github.com/Katsch21/hh2bbtautau/blob/e268752454a0ce0089ff08cc6c373a353be77679/hbt/calibration/tau.py#L117
    """
    assert lepton in ["Muon", "Electron", "Tau", "FatJet", "Jet"], "Lepton not provided"
    
    # get needed lepton and MET fields
    if lepton != "FatJet" and "Jet":
        # get needed lepton and MET fields
        lepton_pt_raw = events[lepton, "pt_raw"]
    else: 
        lepton_pt_raw = events[lepton, "pt_raw_original"]

    #lepton_pt_raw = events[lepton, "pt_raw"]
    lepton_pt = events[lepton, "pt"]
    lepton_phi = events[lepton, "phi"]
    met_pt = events.MET.pt
    met_phi = events.MET.phi
    
    # build px and py sums before and after: we sum the time at x and the time at y of each event    
    old_px = ak.sum(lepton_pt_raw * np.cos(lepton_phi), axis=1)
    old_py = ak.sum(lepton_pt_raw * np.sin(lepton_phi), axis=1)
    new_px = ak.sum(lepton_pt * np.cos(lepton_phi), axis=1)
    new_py = ak.sum(lepton_pt * np.sin(lepton_phi), axis=1)

    # get x and y changes
    delta_x = new_px - old_px
    delta_y = new_py - old_py
    
    # propagate changes to MET (x, y) components: Negative signs have been changed
    met_px = met_pt * np.cos(met_phi) + delta_x
    met_py = met_pt * np.sin(met_phi) + delta_y
    
    # propagate changes to MET (pT, phi) components
    met_pt = np.sqrt((met_px ** 2.0 + met_py ** 2.0))
    met_phi = np.arctan2(met_py, met_px)
    
    # update MET fields
    events["MET", "pt"] = met_pt
    events["MET", "phi"] = met_phi


def add_met_trigger_corrections(
    mask_trigger,
    dataset,
    met: ak.Array,
    weights: Type[Weights],
    year: str,
    variation: str = "nominal",
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
    # Load MET trigger scale factors from correctionlib
    # ------------------------------------------------------------------
    cset = correctionlib.CorrectionSet.from_file(
        f"wprime_plus_b/corrections/met_trigger/met_trigger_{year}_UL.json"
    )

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
        weight_background = "UL-MET-Trigger-SF_WJ"

        sf = cset[weight_background].evaluate(met_pt, "nominal")
        nominal_sf = np.where(mask_trigger, sf, 1.0)

        sf_up = cset[weight_background].evaluate(met_pt, "up")
        sf_down = cset[weight_background].evaluate(met_pt, "down")
        up_sf = np.where(mask_trigger, sf_up, 1.0)
        down_sf = np.where(mask_trigger, sf_down, 1.0)

    elif dataset.startswith("TTTo"):
        # MET trigger SFs for tt̄ background
        weight_background = "UL-MET-Trigger-SF_TT"

        sf = cset[weight_background].evaluate(met_pt, "nominal")
        nominal_sf = np.where(mask_trigger, sf, 1.0)

        sf_up = cset[weight_background].evaluate(met_pt, "up")
        sf_down = cset[weight_background].evaluate(met_pt, "down")
        up_sf = np.where(mask_trigger, sf_up, 1.0)
        down_sf = np.where(mask_trigger, sf_down, 1.0)

    # ------------------------------------------------------------------
    # Store MET trigger scale factors in the weights container
    # ------------------------------------------------------------------
    weights.add(
        name="CMS_eff_MET_trigger",
        weight=nominal_sf,
        weightUp=up_sf,
        weightDown=down_sf,
    )

                

def update_met_jet_veto(events: ak.Array, jets_veto) -> None:
    """
    helper function to compute new MET after lepton pT correction. 
    It uses the 'pt_raw' and 'pt' fields from 'leptons' to update MET 'pt' and 'phi' fields
    
    Parameters:
        - events:
            Events array
        - lepton:
            Lepton name {'Muon', 'Tau'}

    https://github.com/columnflow/columnflow/blob/16d35bb2f25f62f9110a8f1089e8dc5c62b29825/columnflow/calibration/util.py#L42
    https://github.com/Katsch21/hh2bbtautau/blob/e268752454a0ce0089ff08cc6c373a353be77679/hbt/calibration/tau.py#L117
    """
   
    # MET
    met_pt = events.MET.pt
    met_phi = events.MET.phi


    # Jet veto pt(x,y) per event
    jet_veto_pt_x = jets_veto.pt * np.cos(jets_veto.phi)
    jet_veto_pt_y = jets_veto.pt * np.sin(jets_veto.phi)

    # events.Jet.pt
    jet_pt_x = events.Jet.pt * np.cos(events.Jet.phi)
    jet_pt_y = events.Jet.pt * np.sin(events.Jet.phi)
    

    # get x and y changes
    delta_x =  ak.sum(jet_pt_x , axis =-1) - ak.sum(jet_veto_pt_x , axis =-1) 
    delta_y =  ak.sum(jet_pt_y , axis =-1) - ak.sum(jet_veto_pt_y , axis =-1) 

    
    # propagate changes to MET (x, y) components:Problematic samples y TTToSemiLeptonic and SingleMuon
    met_px = met_pt * np.cos(met_phi) - delta_x
    met_py = met_pt * np.sin(met_phi) - delta_y
    

    # propagate changes to MET (pT, phi) components
    new_met_pt = np.sqrt((met_px ** 2.0 + met_py ** 2.0))
    new_met_phi = np.arctan2(met_py, met_px)
    
    # update MET fields
    events["MET", "pt"] = new_met_pt
    events["MET", "phi"] = new_met_phi



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


