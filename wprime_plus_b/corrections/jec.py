import gzip
import cloudpickle
import numpy as np
import awkward as ak
import importlib.resources
from typing import Tuple
from coffea.nanoevents.methods.base import NanoEventsArray
from wprime_plus_b.corrections.met import update_met, update_met_list


# Recomendations https://twiki.cern.ch/twiki/bin/viewauth/CMS/JECDataMC#Recommended_for_MC
def apply_jet_corrections(events: NanoEventsArray, year: str, variation: bool) -> None:
    """
    Apply JEC/JER corrections to jets (propagate to MET)

    We use the script data/scripts/build_jec.py to create the 'mc_jec_compiled.pkl.gz'
    file with jet and MET factories

    Parameters:
    -----------
        events:
            events collection
        year:
            Year of the dataset {'2016APV', '2016', '2017', '2018'}
    """
    # load jet and MET factories with JEC/JER corrections
    with importlib.resources.path(
        "wprime_plus_b.data", "mc_jec_compiled.pkl.gz"
    ) as path:
        with gzip.open(path) as fin:
            factories = cloudpickle.load(fin)

        
    def add_jec_variables(jets: ak.Array, event_rho: ak.Array):
        """add some variables to the jet collection"""
#        jets["pt_raw_original"] = jets.pt
        jets["pt_raw"] = (1 - jets.rawFactor) * jets.pt
        jets["mass_raw"] = (1 - jets.rawFactor) * jets.mass
        jets["pt_gen"] = ak.values_astype(
            ak.fill_none(jets.matched_gen.pt, 0), np.float32
        )
        jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]
        return jets

    # get corrected jets
    events["Jet"] = factories["jet_factory"][year].build(
        add_jec_variables(events.Jet, events.fixedGridRhoFastjetAll),
        events.caches[0],
    )

    # get corrected MET
    events["MET"] = factories["met_factory"].build(events.MET, events.Jet, {})
    events["MET", "Unclustered_nom_pt"] = events.MET.pt  
    

    if variation:

        met_pt_list = {
            "Jet_JES": {
                "up": events.MET.JES_jes.up.pt,
                "nom": events.MET.pt,
                "down": events.MET.JES_jes.down.pt,
            },
            
            "Jet_JER": {
                "up": events.MET.JER.up.pt,
                "nom": events.MET.pt,
                "down": events.MET.JER.down.pt
            }

        }


        met_phi_list = {
            "Jet_JES": {
                "up": events.MET.phi,
                "nom": events.MET.phi,
                "down": events.MET.phi,
            },
            
            "Jet_JER": {
                "up": events.MET.phi,
                "nom": events.MET.phi,
                "down": events.MET.phi
            }
        }

        delta_list = {
            "Jet_JES": {
                "delta_x": {
                    "nom": met_pt_list["Jet_JES"]["nom"] * np.cos(met_phi_list["Jet_JES"]["nom"]) - events.MET.pt_orig * np.cos(events.MET.phi_orig),
                    "up": met_pt_list["Jet_JES"]["up"] * np.cos(met_phi_list["Jet_JES"]["up"]) - events.MET.pt_orig * np.cos(events.MET.phi_orig),
                    "down": met_pt_list["Jet_JES"]["down"] * np.cos(met_phi_list["Jet_JES"]["down"]) - events.MET.pt_orig * np.cos(events.MET.phi_orig)
                },
                "delta_y": {
                    "nom": met_pt_list["Jet_JES"]["nom"] * np.sin(met_phi_list["Jet_JES"]["nom"]) -  events.MET.pt_orig * np.sin(events.MET.phi_orig),
                    "up": met_pt_list["Jet_JES"]["up"] * np.sin(met_phi_list["Jet_JES"]["up"]) -  events.MET.pt_orig * np.sin(events.MET.phi_orig),
                    "down": met_pt_list["Jet_JES"]["down"] * np.sin(met_phi_list["Jet_JES"]["down"]) -  events.MET.pt_orig * np.sin(events.MET.phi_orig)
                },                
            },

            "Jet_JER": {
                "delta_x": {
                    "nom": met_pt_list["Jet_JER"]["nom"] * np.cos(met_phi_list["Jet_JER"]["nom"]) - events.MET.pt_orig * np.cos(events.MET.phi_orig),
                    "up": met_pt_list["Jet_JER"]["up"] * np.cos(met_phi_list["Jet_JER"]["up"]) - events.MET.pt_orig * np.cos(events.MET.phi_orig),
                    "down": met_pt_list["Jet_JER"]["down"] * np.cos(met_phi_list["Jet_JER"]["down"]) - events.MET.pt_orig * np.cos(events.MET.phi_orig)
                },
                "delta_y": {
                    "nom": met_pt_list["Jet_JER"]["nom"] * np.sin(met_phi_list["Jet_JER"]["nom"]) -  events.MET.pt_orig * np.sin(events.MET.phi_orig),
                    "up": met_pt_list["Jet_JER"]["up"] * np.sin(met_phi_list["Jet_JER"]["up"]) -  events.MET.pt_orig * np.sin(events.MET.phi_orig),
                    "down": met_pt_list["Jet_JER"]["down"] * np.sin(met_phi_list["Jet_JER"]["down"]) -  events.MET.pt_orig * np.sin(events.MET.phi_orig)
                }, 
        },

            "MET_uncluster": {
                "delta_x": {
                    "nom":  events.MET.Unclustered_nom_pt * np.cos(events.MET.phi_orig) - events.MET.pt_orig * np.cos(events.MET.phi_orig),
                    "up": events.MET.MET_UnclusteredEnergy.up.pt * np.sin(events.MET.MET_UnclusteredEnergy.up.phi) - events.MET.pt_orig * np.sin(events.MET.phi_orig),
                    "down": events.MET.MET_UnclusteredEnergy.up.pt * np.sin(events.MET.MET_UnclusteredEnergy.up.phi) - events.MET.pt_orig * np.sin(events.MET.phi_orig)
                },
                "delta_y": {
                    "nom": events.MET.Unclustered_nom_pt * np.sin(events.MET.phi_orig) - events.MET.pt_orig * np.sin(events.MET.phi_orig),
                    "up": events.MET.MET_UnclusteredEnergy.down.pt * np.sin(events.MET.MET_UnclusteredEnergy.down.phi) - events.MET.pt_orig * np.sin(events.MET.phi_orig),
                    "down": events.MET.MET_UnclusteredEnergy.down.pt  *  np.sin(events.MET.MET_UnclusteredEnergy.down.phi)- events.MET.pt_orig * np.sin(events.MET.phi_orig),
                },
            },
        }


      
        return delta_list #met_pt_list, met_phi_list, delta_list




# FatJet: JER and JEC
def apply_fatjet_corrections(events: NanoEventsArray, year: str, variation: bool = False, delta_variation_list: list = None) -> None:
    """
    Apply JEC/JER corrections to jets (propagate to MET)

    We use the script data/scripts/build_jec.py to create the 'mc_jec_compiled.pkl.gz'
    file with jet and MET factories

    Parameters:
    -----------
        events:
            events collection
        year:
            Year of the dataset {'2016APV', '2016', '2017', '2018'}
    """

    # load jet and MET factories with JEC/JER corrections
    with importlib.resources.path(
        "wprime_plus_b.data", "mc_jec_compiled.pkl.gz"
    ) as path:
        with gzip.open(path) as fin:
            factories = cloudpickle.load(fin)


    def add_jec_variables(fatjets: ak.Array, event_rho: ak.Array):
        """add some variables to the jet collection"""
        fatjets["pt_raw_original"] = fatjets.pt
        fatjets["pt_raw"] = (1 - fatjets.rawFactor) * fatjets.pt
        fatjets["mass_raw"] = (1 - fatjets.rawFactor) * fatjets.mass
        fatjets["pt_gen"] = ak.values_astype(
            ak.fill_none(fatjets.matched_gen.pt, 0), np.float32
        )
        fatjets["event_rho"] = ak.broadcast_arrays(event_rho, fatjets.pt)[0]
        return fatjets

    # get corrected fatjets
    events["FatJet"] = factories["fatjet_factory"][year].build(
        add_jec_variables(events.FatJet, events.fixedGridRhoFastjetAll),
        events.caches[0],
    )

    if variation:

        # Fatjet mask
        fatjet_mask = ak.num(events.FatJet) > 0
        
        fatjet_list = {
            "FAT_JES": {
                "nom": events.FatJet.pt, 
                "up": events.FatJet.JES_jes.up.pt, 
                "down": events.FatJet.JES_jes.down.pt, 
                "phi": events.FatJet.phi, 
                "raw": events.FatJet.pt_orig, 
            },
            "FAT_JER": {
                "nom": events.FatJet.pt, 
                "up": events.FatJet.JER.up.pt, 
                "down": events.FatJet.JER.down.pt, 
                "phi": events.FatJet.phi, 
                "raw": events.FatJet.pt_orig,   
            },
        }
        

        delta_list_combined = {}
        
        met_pt_list, met_phi_list, delta_list_combined_jes = update_met_list(   
            events=events,
            syst_name="FatJet_JES",
            syst_var = variation, 
            object = fatjet_list["FAT_JES"],    
        )

        met_pt_list, met_phi_list, delta_list_combined_jer = update_met_list(   
            events=events,
            syst_name="FatJet_JER",
            syst_var = variation, 
            object = fatjet_list["FAT_JER"],  
        )

        delta_list_combined.update(delta_list_combined_jes)
        delta_list_combined.update(delta_list_combined_jer)

        
        return delta_list_combined #met_pt_list, met_phi_list, delta_list

        

    else:
        # propagate corrections to MET
        update_met(events=events, lepton="FatJet")
