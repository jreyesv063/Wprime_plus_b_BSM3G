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
    topjets: ak.Array,
    weights: Type[Weights],
    year: str = "2017",
    working_point_topjet: str = "Tight",
    top_mask = None
):
    """
    Adds event-level weights comparing Top-jet vs QCD efficiencies using ParticleNet TvsQCD tagger scale factors.
    The function flattens the input fatjet array, selects good Top-jet candidates within pt/eta and working-point limits,
    evaluates the corresponding correction set (nominal, up, down), restores the original jet structure (unflatten),
    and stores the resulting weights in the provided Weights container.
    """
    # ===========================================================
    #  Read json files
    # ============================================================
    # Correction name
    with open("wprime_plus_b/corrections/correction_names/JME.json", "r") as f:
        case = json.load(f)

    correction_name = case['CMS_eff_j_ParticleNet_Top_Nominal'][year]
        
    # Wps top tagger
    with open("wprime_plus_b/json_files/fatjet.json", "r") as f: 
        Wps = json.load(f)

    # Particle Net: Working points        
    pNet_id = Wps[year]["TvsQCD"][working_point_topjet]['value']
    mistagging_rate =  Wps[year]["TvsQCD"][working_point_topjet]['mistagging_rate']
    

    # =============================================================
    #  Fat jet candidates
    # =============================================================    
    # flat fatjets array since correction function works only on flat arrays
    tj, n = ak.flatten(topjets), ak.num(topjets)
    
    # get 'in-limits' jets
    topjet_pt_mask = ((tj.pt >= 300.0) & (tj.pt < 1200.0))
    
    if year == "2017" or year == "2018":
        topjet_eta_mask = ((np.abs(tj.eta) < 2.499))
    else:
        topjet_eta_mask = ((np.abs(tj.eta) < 2.399)) 

    # Passing working point
    topjet_wp_mask = ((tj.particleNet_TvsQCD >= pNet_id))

    
    in_topjet_mask = topjet_pt_mask & topjet_eta_mask & topjet_wp_mask
    
    in_topjets = tj.mask[in_topjet_mask]
    
    # get jet transverse momentum and pseudorapidity (replace None values with some 'in-limit' value)
    topjets_pt = ak.fill_none(in_topjets.pt, 400.0)
    topjets_eta = ak.fill_none(in_topjets.eta, 0.0)
    
    # =============================================================
    # Correction: event-level weight (nominal/up/down)
    # =============================================================
    cset = correctionlib.CorrectionSet.from_file(get_pog_json("pujetid", year))
    
    # Get nominal, up, and down scale factors: If jet in 'in-limits' jets, then take the computed SF, otherwise assign 1
    nominal_sf, up_sf, down_sf = [
        unflat_sf(cset[correction_name].evaluate(topjets_eta, topjets_pt, v, mistagging_rate), in_topjet_mask, n)
        for v in ("nom", "up", "down")
    ]
    
    nominal_sf, up_sf, down_sf = [
        ak.where(top_mask, sf, 1.0)
        for sf in (nominal_sf, up_sf, down_sf)
    ]

    # add nominal, up and down scale factors to weights container
    weights.add(
        name=f"CMS_eff_j_ParticleNet_Top_Nominal_{year}",
        weight=nominal_sf,
        weightUp=up_sf,
        weightDown=down_sf,
    )
    
    
# ParticleNet_W_Nominal
def add_QCD_vs_W_weight(
    wjets: ak.Array,
    weights: Type[Weights],
    year: str = "2017",
    working_point_wjet: str = "Tight",
    W_mask = None
):
    """
    Adds event-level weights comparing W-jet vs QCD efficiencies using ParticleNet WvsQCD tagger scale factors.
    The function flattens the input jet array, selects good W-jet candidates within pt/eta and working-point limits,
    evaluates the corresponding correction set (nominal, up, down), restores the original shape (unflatten),
    and stores the resulting weights in the provided Weights container.
    """
    # ===========================================================
    #  Read json files
    # ============================================================
    # Correction name
    with open("wprime_plus_b/corrections/correction_names/JME.json", "r") as f:
        case = json.load(f)

    correction_name = case['CMS_eff_j_ParticleNet_W_Nominal'][year]
        
    # Wps top tagger
    with open("wprime_plus_b/json_files/fatjet.json", "r") as f: 
        Wps = json.load(f)

    # Particle Net: Working points        
    pNet_id = Wps[year]["WvsQCD"][working_point_wjet]['value']
    mistagging_rate =  Wps[year]["WvsQCD"][working_point_wjet]['mistagging_rate']
    

    # =============================================================
    #  W jet candidates
    # =============================================================    
    # flat fatjets array since correction function works only on flat arrays
    wj, n = ak.flatten(wjets), ak.num(wjets)
    
    # get 'in-limits' jets
    wjet_pt_mask = ((wj.pt >= 200.0) & (wj.pt < 800.0))
    
    if year == "2017" or year == "2018":
        wjet_eta_mask = ((np.abs(wj.eta) < 2.499))
    else:
        wjet_eta_mask = ((np.abs(wj.eta) < 2.399)) 

    # Passing working point
    wjet_wp_mask = ((wj.particleNet_WvsQCD >= pNet_id))

    
    in_wjet_mask = wjet_pt_mask & wjet_eta_mask & wjet_wp_mask
    
    in_wjets = wj.mask[in_wjet_mask]
    
    # get jet transverse momentum and pseudorapidity (replace None values with some 'in-limit' value)
    wjets_pt = ak.fill_none(in_wjets.pt, 400.0)
    wjets_eta = ak.fill_none(in_wjets.eta, 0.0)
    
    # =============================================================
    # Correction: event-level weight (nominal/up/down)
    # =============================================================
    cset = correctionlib.CorrectionSet.from_file(get_pog_json("pujetid", year))
    
    # Get nominal, up, and down scale factors: If jet in 'in-limits' jets, then take the computed SF, otherwise assign 1
    nominal_sf, up_sf, down_sf = [
        unflat_sf(cset[correction_name].evaluate(wjets_eta, wjets_pt, v, mistagging_rate), in_wjet_mask, n)
        for v in ("nom", "up", "down")
    ]
    
    nominal_sf, up_sf, down_sf = [
        ak.where(W_mask, sf, 1.0)
        for sf in (nominal_sf, up_sf, down_sf)
    ]
        
    # add nominal, up and down scale factors to weights container
    weights.add(
        name=f"CMS_eff_j_ParticleNet_W_Nominal_{year}",
        weight=nominal_sf,
        weightUp=up_sf,
        weightDown=down_sf,
    )
