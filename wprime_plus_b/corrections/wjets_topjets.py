import correctionlib
import numpy as np
import json
import awkward as ak
from typing import Type
from .utils import unflat_sf
from coffea.analysis_tools import Weights
from wprime_plus_b.corrections.utils import get_pog_json

"""
add jet ParticleNet_Top_Nominal scale factor and ParticleNet_W_Nominal SFs

Parameters:
-----------
    eta: [-2.5, 2.5)

    pt:  Tops -> [300, 1200);  Ws -> [200, 800)

    systematics: down, nom, up

    workingpoint: Tops -> 0p1; 0p5; 1p0; Ws -> 0p5; 1p0; 5p0
    
    https://twiki.cern.ch/twiki/bin/view/CMS/ParticleNetTopWSFs
    
            https://indico.physics.lbl.gov/event/975/contributions/8301/attachments/4047/5437/23.07.31_BOOST_Xbbcc_performance_CL.pdf

"""
    
# ParticleNet_Top_Nominal
def add_QCD_vs_Top_weight(
    fatjets: ak.Array,
    weights: Type[Weights],
    year: str = "2017",
    year_mod: str = "",
    working_point_fatjet: str = "Tight",
    variation: str = "nominal",
):
    
    # Please, check the table https://twiki.cern.ch/twiki/bin/viewauth/CMS/ParticleNetTopWSFs (2017 Data). You will understand the meaning of 0p1; 5p0; etc.
    mistagging_rate_equivalence = {
        "Loose": "1p0",
        "Medium": "0p5",
        "Tight": "0p1"
    }
    
    
    # Wps top tagger
    with open("wprime_plus_b/jsons/topWps.json", "r") as f: 
        Wps = json.load(f)
        
    pNet_id = Wps[year + year_mod]["TvsQCD"][working_point_fatjet]  
    
    
    # flat fatjets array since correction function works only on flat arrays
    fj, n = ak.flatten(fatjets), ak.num(fatjets)
    
    # get 'in-limits' jets
    fatjet_pt_mask = (
        (fj.pt >= 300.0) 
        & (fj.pt < 1200.0)
    )
    
    if year == "2017" or year == "2018":
        fatjet_eta_mask = (
            (np.abs(fj.eta) < 2.499)
        )

    else:
        fatjet_eta_mask = (
            (np.abs(fj.eta) < 2.399)
        )

    fatjet_wp_mask = (
        (fj.particleNet_TvsQCD >= pNet_id)
    )
    
    in_fatjet_mask = fatjet_pt_mask & fatjet_eta_mask & fatjet_wp_mask
    
    in_fatjets = fj.mask[in_fatjet_mask]
    
    # get jet transverse momentum and pseudorapidity (replace None values with some 'in-limit' value)
    fatjets_pt = ak.fill_none(in_fatjets.pt, 400.0)
    fatjets_eta = ak.fill_none(in_fatjets.eta, 0.0)
    working_point = mistagging_rate_equivalence[working_point_fatjet]
    
    
    # define correction set
    cset = correctionlib.CorrectionSet.from_file(
        get_pog_json("pujetid", year + year_mod)
    )
    
    # get nominal scale factors
    # If jet in 'in-limits' jets, then take the computed SF, otherwise assign 1
    # Unflatten to original shape
    nominal_sf = unflat_sf(
        cset["ParticleNet_Top_Nominal"].evaluate(fatjets_eta, fatjets_pt, "nom", working_point),
        in_fatjet_mask,
        n,
    )
    
    if variation == "nominal":
        # get 'up' and 'down' variations
        up_sf = unflat_sf(
            cset["ParticleNet_Top_Nominal"].evaluate(fatjets_eta, fatjets_pt, "up", working_point),
            in_fatjet_mask,
            n,
        )
        down_sf = unflat_sf(
            cset["ParticleNet_Top_Nominal"].evaluate(fatjets_eta, fatjets_pt, "down", working_point),
            in_fatjet_mask,
            n,
        )
        # add nominal, up and down scale factors to weights container
        weights.add(
            name="QCD_vs_Top",
            weight=nominal_sf,
            weightUp=up_sf,
            weightDown=down_sf,
        )
    else:
        # add nominal scale factors to weights container
        weights.add(name="QCD_vs_Top", weight=nominal_sf)    
    

    
# ParticleNet_W_Nominal
def add_QCD_vs_W_weight(
    wjets: ak.Array,
    weights: Type[Weights],
    year: str = "2017",
    year_mod: str = "",
    working_point_wjet: str = "Tight",
    variation: str = "nominal",
):
    
    # Please, check the table https://twiki.cern.ch/twiki/bin/viewauth/CMS/ParticleNetTopWSFs (2017 Data). You will understand the meaning of 0p1; 5p0; etc.
    mistagging_rate_equivalence = {
        "Loose": "5p0",
        "Medium": "1p0",
        "Tight": "0p5"
    }
    
    
    # Wps top tagger
    with open("wprime_plus_b/jsons/topWps.json", "r") as f: 
        Wps = json.load(f)
        
    pNet_id = Wps[year + year_mod]["WvsQCD"][working_point_wjet]  
    
    
    # flat fatjets array since correction function works only on flat arrays
    wj, n = ak.flatten(wjets), ak.num(wjets)
    
    # get 'in-limits' jets
    wjet_pt_mask = (
        (wj.pt >= 200.0) 
        & (wj.pt < 800.0)
    )
    
    if year == "2017" or year == "2018":
        wjet_eta_mask = (
            (np.abs(wj.eta) < 2.499)
        )
    else:
        wjet_eta_mask = (
            (np.abs(wj.eta) < 2.399)
        ) 
    
    wjet_wp_mask = (
        (wj.particleNet_WvsQCD >= pNet_id)
    )
    
    in_wjet_mask = wjet_pt_mask & wjet_eta_mask & wjet_wp_mask
    
    in_wjets = wj.mask[in_wjet_mask]
    
    # get jet transverse momentum and pseudorapidity (replace None values with some 'in-limit' value)
    wjets_pt = ak.fill_none(in_wjets.pt, 400.0)
    wjets_eta = ak.fill_none(in_wjets.eta, 0.0)
    working_point = mistagging_rate_equivalence[working_point_wjet]
    
    
    # define correction set
    cset = correctionlib.CorrectionSet.from_file(
        get_pog_json("pujetid", year + year_mod)
    )
    
    # get nominal scale factors
    # If jet in 'in-limits' jets, then take the computed SF, otherwise assign 1
    # Unflatten to original shape
    nominal_sf = unflat_sf(
        cset["ParticleNet_W_Nominal"].evaluate(wjets_eta, wjets_pt, "nom", working_point),
        in_wjet_mask,
        n,
    )
    
    if variation == "nominal":
        # get 'up' and 'down' variations
        up_sf = unflat_sf(
            cset["ParticleNet_W_Nominal"].evaluate(wjets_eta, wjets_pt, "up", working_point),
            in_wjet_mask,
            n,
        )
        down_sf = unflat_sf(
            cset["ParticleNet_W_Nominal"].evaluate(wjets_eta, wjets_pt, "down", working_point),
            in_wjet_mask,
            n,
        )
        # add nominal, up and down scale factors to weights container
        weights.add(
            name="QCD_vs_W",
            weight=nominal_sf,
            weightUp=up_sf,
            weightDown=down_sf,
        )
    else:
        # add nominal scale factors to weights container
        weights.add(name="QCD_vs_W", weight=nominal_sf)    
    
