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
) -> ak.highlevel.Array:
    
    # Wps top tagger, jet_id and jet_eta
    with open("wprime_plus_b/jsons/topWps.json", "r") as f: 
        Wps = json.load(f)
 
    pNet_id = Wps[year]["TvsQCD"][TvsQCD]                      
    jet_id = Wps[year]['jet_id']  
    
    return (
                (fatjets.pt >= fatjet_pt_threshold)
                & (np.abs(fatjets.eta) <= fatjet_eta_threshold)
                & (fatjets.particleNet_TvsQCD >= pNet_id)   
                & (fatjets.jetId >= jet_id)                   
    )