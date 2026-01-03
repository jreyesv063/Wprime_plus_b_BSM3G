import numpy as np
import awkward as ak
from typing import List, Optional, Dict, Any

def histograms_output_array(
    self_main: Any,
    njets_no_top: Optional[Dict[str, ak.Array]] = None,
    tops: Optional[ak.Array] = None,
    objects: Optional[list] = None,
    mask: Optional[ak.Array] = None,
    name: str = "",
    syst_name: str = "",
    lepton_flavor: str = "tau"
):


    # Select region objects
    region_jets = objects["jets_veto"][mask]  
    region_bjets = objects["bjets"][mask]
    region_lightjets = objects["jets"][mask]  
    region_fatjets = objects["fatjets"][mask]
    region_wjets = objects["wjets"][mask]   

    region_tops = tops[mask]

    region_electrons = objects["electrons"][mask]
    region_muons = objects["muons"][mask]
    region_taus = objects["taus"][mask]
    region_met = objects["met"][mask]


    # Define region leptons
    lepton_region_map = {
        "ele": region_electrons,
        "mu": region_muons,
        "tau": region_taus
    }


    region_leptons = lepton_region_map[lepton_flavor]

    # =================================================
    #             Main obs: m_T(lepton,MET)
    # =================================================
    lepton_met_mass = np.sqrt(
        2.0
        * region_leptons.pt
        * region_met.pt
        * (
            ak.ones_like(region_met.pt)
            - np.cos(region_leptons.delta_phi(region_met))
        )
    )

    # =================================================
    #        Delta-phi (lepton, met)
    # =================================================
    delta_phi_lepton_met = region_leptons.delta_phi(region_met)
    
    # =================================================
    #  Pt per event
    # =================================================
    lightjet_pt_addition = ak.sum(region_lightjets.pt, axis=1)
    bjet_pt_addition = ak.sum(region_bjets.pt, axis=1)


    fatjet_pt_addition = ak.sum(region_fatjets.pt, axis=1)
    wjet_pt_addition = ak.sum(region_wjets.pt, axis=1)   
    
    lepton_pt_addition =  ak.sum(region_leptons.pt, axis = 1)


    # =================================================
    #    HT ans ST
    # =================================================      
    region_HT = lightjet_pt_addition  + bjet_pt_addition + fatjet_pt_addition + wjet_pt_addition
    region_ST = region_HT + lepton_pt_addition
    region_ST_met = region_ST + region_met.pt 
        
    if njets_no_top is not None:
        region_counts = {
            key: array[mask]
            for key, array in njets_no_top.items()
        }
        self_main.add_feature(f"njets_no_top_tagger", region_counts["njets_no_top"]+ region_counts["nbjets_no_top"] + region_counts["nfatjets_no_top"] + region_counts["nwjets_no_top"])


    if syst_name == "":
        self_main.add_feature(f"lepton_pt_{name}", region_leptons.pt)
        self_main.add_feature(f"lepton_eta_{name}", region_leptons.eta)
        self_main.add_feature(f"lepton_phi_{name}", region_leptons.phi)

        # Bjets
        self_main.add_feature(f"bjet_pt_{name}", ak.firsts(region_bjets).pt)
        self_main.add_feature(f"bjet_eta_{name}",  ak.firsts(region_bjets).eta)
        self_main.add_feature(f"bjet_phi_{name}",  ak.firsts(region_bjets).phi)

        # MET
        self_main.add_feature(f"met_{name}", region_met.pt)
        self_main.add_feature(f"met_phi_{name}", region_met.phi)
        self_main.add_feature(f"recoil_pt_{name}", region_met.pt_recoil)

        
        # Transverse mass and delta_phi: lepton; met.   
        self_main.add_feature(f"lepton_met_mass_{name}", lepton_met_mass)
        self_main.add_feature(f"lepton_met_phi_{name}", delta_phi_lepton_met)


        # Number of objects
        self_main.add_feature(f"njets_full_{name}", ak.num(region_lightjets) + ak.num(region_bjets) +  ak.num(region_fatjets) + ak.num(region_wjets))
        self_main.add_feature(f"npvs_{name}", objects["events"].PV.npvsGood[mask])
        self_main.add_feature(f"nmuons_{name}", ak.num(region_muons))
        self_main.add_feature(f"nelectrons_{name}", ak.num(region_electrons))
        self_main.add_feature(f"ntaus_{name}", ak.num(region_taus))
        self_main.add_feature(f"nbjets_{name}", ak.num(region_bjets))


        # Scalar sum of transverse momenta
        self_main.add_feature(f"HT_{name}", region_HT) 
        self_main.add_feature(f"ST_{name}", region_ST)  
        self_main.add_feature(f"ST_met_{name}", region_ST_met)          

        # Top reconstructed mass
        self_main.add_feature(f"top_mrec_{name}", region_tops)

    else:
        self_main.add_feature(f"lepton_met_mass_{name}_{syst_name}", lepton_met_mass)