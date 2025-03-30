import json
import numpy as np
import awkward as ak
import importlib.resources

def select_good_fatjets(
    fatjets,
    year: str = "2017",
    fatjet_pt_threshold: float = 300.0,
    fatjet_eta_threshold: float = 2.4,
    TvsQCD: str = "Tight",
    is_mc: bool = True,
) -> ak.highlevel.Array:
    
    # Wps top tagger, jet_id and jet_eta
    with open("wprime_plus_b/jsons/topWps.json", "r") as f: 
        Wps = json.load(f)
 
    pNet_id = Wps[year]["TvsQCD"][TvsQCD]                      
    jet_id = Wps[year]['jet_id']  
    
    good_fatjet_masks = {}

    # Si no hay fatjets, devolver máscaras vacías
    if len(fatjets) == 0:
        good_fatjet_masks["nominal"] = ak.zeros_like(fatjets.pt, dtype=bool)
        if is_mc:
            good_fatjet_masks["JES_up"] = ak.zeros_like(fatjets.pt, dtype=bool)
            good_fatjet_masks["JES_down"] = ak.zeros_like(fatjets.pt, dtype=bool)
            good_fatjet_masks["JER_up"] = ak.zeros_like(fatjets.pt, dtype=bool)
            good_fatjet_masks["JER_down"] = ak.zeros_like(fatjets.pt, dtype=bool)
        return good_fatjet_masks

    # Si no tiene el atributo JES_jes, manejarlo adecuadamente
    if is_mc and not hasattr(fatjets, "JES_jes"):
        good_fatjet_masks["nominal"] = (
            (fatjets.pt >= fatjet_pt_threshold)
            & (np.abs(fatjets.eta) <= fatjet_eta_threshold)
            & (fatjets.particleNet_TvsQCD >= pNet_id)   
            & (fatjets.jetId >= jet_id)
        )
        good_fatjet_masks["JES_up"] = ak.zeros_like(fatjets.pt, dtype=bool)
        good_fatjet_masks["JES_down"] = ak.zeros_like(fatjets.pt, dtype=bool)
        good_fatjet_masks["JER_up"] = ak.zeros_like(fatjets.pt, dtype=bool)
        good_fatjet_masks["JER_down"] = ak.zeros_like(fatjets.pt, dtype=bool)
        return good_fatjet_masks

    # Procesar fatjets para MC
    if is_mc:
        jet_shift = {
            "JES": {"nominal": fatjets,
                    "up":  fatjets.JES_jes.up,
                    "down": fatjets.JES_jes.down
            },
            "JER": {
                    "up":  fatjets.JER.up,
                    "down": fatjets.JER.down
            },  
        }

        for shift_type, variations in jet_shift.items():
            for variation, shift in variations.items():
                
                good_fatjet = (
                    (shift.pt >= fatjet_pt_threshold)
                    & (np.abs(shift.eta) <= fatjet_eta_threshold)
                    & (shift.particleNet_TvsQCD >= pNet_id)   
                    & (shift.jetId >= jet_id)    
                )

                if variation == "nominal":
                    good_fatjet_masks[variation] = good_fatjet
                else:
                    good_fatjet_masks[f"{shift_type}_{variation}"] = good_fatjet

    # Procesar fatjets para datos
    else:
        good_fatjet = (
            (fatjets.pt >= fatjet_pt_threshold)
            & (np.abs(fatjets.eta) <= fatjet_eta_threshold)
            & (fatjets.particleNet_TvsQCD >= pNet_id)   
            & (fatjets.jetId >= jet_id)                   
        )
        good_fatjet_masks["nominal"] = good_fatjet

    return good_fatjet_masks

    """
    good_fatjet_masks = {}

    if is_mc:
        jet_shift = {
            "JES": {"nominal": fatjets,
                    "up":  fatjets.JES_jes.up,
                    "down": fatjets.JES_jes.down
            },
            "JER": {
                    "up":  fatjets.JER.up,
                    "down": fatjets.JER.down
            },  
        }

        for shift_type, variations in jet_shift.items():
            for variation, shift in variations.items():
                
                good_fatjet = (
                    (shift.pt >= fatjet_pt_threshold)
                    & (np.abs(shift.eta) <= fatjet_eta_threshold)
                    & (shift.particleNet_TvsQCD >= pNet_id)   
                    & (shift.jetId >= jet_id)    
                )

                if variation == "nominal":
                    good_fatjet_masks[variation] = good_fatjet
                else:
                    good_fatjet_masks[f"{shift_type}_{variation}"] = good_fatjet

             
    else:
        good_fatjet = (
                (fatjets.pt >= fatjet_pt_threshold)
                & (np.abs(fatjets.eta) <= fatjet_eta_threshold)
                & (fatjets.particleNet_TvsQCD >= pNet_id)   
                & (fatjets.jetId >= jet_id)                   
        )
        good_fatjet_masks["nominal"] = good_fatjet

    return good_fatjet_masks"
    """