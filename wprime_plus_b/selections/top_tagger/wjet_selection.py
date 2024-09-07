import json
import numpy as np
import awkward as ak
import importlib.resources


def select_good_wjets(
    wjets, 
    year="2017", 
    w_pt_threshold: float = 200.0,
    w_eta_threshold: float = 2.4,
    WvsQCD: str = "Tight"
) -> ak.highlevel.Array:

    
    # Wps top tagger, jet_id and jet_eta
    with open("wprime_plus_b/jsons/topWps.json", "r") as f: 
        Wps = json.load(f)
        
    pNet_id = Wps[year]["WvsQCD"][WvsQCD]                      
    jet_id = Wps[year]['jet_id']  
    
        
    return (
                (wjets.pt >= w_pt_threshold)
                & (np.abs(wjets.eta) <= w_eta_threshold)
                & (wjets.particleNet_WvsQCD >= pNet_id)   # W vs QCD (tight) 
                & (wjets.jetId >= jet_id)   
    )
