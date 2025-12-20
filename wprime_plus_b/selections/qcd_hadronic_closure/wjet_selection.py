import json
import numpy as np
import awkward as ak
import importlib.resources


def select_good_wjets(
    wjets, 
    year="2017", 
    w_pt_threshold: float = 200.0,
    w_eta_threshold: float = 2.4,
    WvsQCD: str = "Tight",
    is_mc: bool = True,
) -> ak.highlevel.Array:

    
    # Wps top tagger, jet_id and jet_eta
    with open("wprime_plus_b/jsons/topWps.json", "r") as f: 
        Wps = json.load(f)
        
    pNet_id = Wps[year]["WvsQCD"][WvsQCD]                      
    jet_id = Wps[year]['jet_id']  
    

    good_wjet_masks = {}


    # Si no hay wjets, devolver máscaras vacías
    if len(wjets) == 0:
        good_wjet_masks["nominal"] = ak.zeros_like(wjets.pt, dtype=bool)
        if is_mc:
            good_wjet_masks["JES_up"] = ak.zeros_like(wjets.pt, dtype=bool)
            good_wjet_masks["JES_down"] = ak.zeros_like(wjets.pt, dtype=bool)
            good_wjet_masks["JER_up"] = ak.zeros_like(wjets.pt, dtype=bool)
            good_wjet_masks["JER_down"] = ak.zeros_like(wjets.pt, dtype=bool)
        return good_wjet_masks

    # Si no tiene el atributo JES_jes, manejarlo adecuadamente
    if is_mc and not hasattr(wjets, "JES_jes"):
        good_wjet_masks["nominal"] = (
            (wjets.pt >= w_pt_threshold)
            & (np.abs(wjets.eta) <= w_eta_threshold)
            & (wjets.particleNet_WvsQCD >= pNet_id)   
            & (wjets.jetId >= jet_id)
        )
        good_wjet_masks["JES_up"] = ak.zeros_like(wjets.pt, dtype=bool)
        good_wjet_masks["JES_down"] = ak.zeros_like(wjets.pt, dtype=bool)
        good_wjet_masks["JER_up"] = ak.zeros_like(wjets.pt, dtype=bool)
        good_wjet_masks["JER_down"] = ak.zeros_like(wjets.pt, dtype=bool)
        return good_wjet_masks

    # Procesar wjets para MC
    if is_mc:
        jet_shift = {
            "JES": {"nominal": wjets,
                    "up":  wjets.JES_jes.up,
                    "down": wjets.JES_jes.down
            },
            "JER": {
                    "up":  wjets.JER.up,
                    "down": wjets.JER.down
            },  
        }

        for shift_type, variations in jet_shift.items():
            for variation, shift in variations.items():
                
                good_wjet = (
                    (shift.pt >= w_pt_threshold)
                    & (np.abs(shift.eta) <= w_eta_threshold)
                    & (shift.particleNet_WvsQCD >= pNet_id)     # W vs QCD (tight) 
                    & (shift.jetId >= jet_id)    
                )

                if variation == "nominal":
                    good_wjet_masks[variation] = good_wjet
                else:
                    good_wjet_masks[f"{shift_type}_{variation}"] = good_wjet

    # Procesar wjets para datos
    else:
        good_wjet = (
            (wjets.pt >= w_pt_threshold)
            & (np.abs(wjets.eta) <= w_eta_threshold)
            & (wjets.particleNet_WvsQCD >= pNet_id)   
            & (wjets.jetId >= jet_id)                   
        )
        good_wjet_masks["nominal"] = good_wjet

    return good_wjet_masks