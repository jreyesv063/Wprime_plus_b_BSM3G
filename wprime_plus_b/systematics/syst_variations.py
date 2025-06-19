import numpy as np
import awkward as ak


from wprime_plus_b.processors.utils.analysis_utils import delta_r_mask
# Systematics: Object - level
from wprime_plus_b.systematics.utils import one_var_per_event_MET


def systematic_variation_mask(events, jets_veto, 
                                electrons, muons, taus, bjets, jets, fatjets, wjets,
                                muons_mask, taus_mask, bjets_mask, light_jets_mask, fatjets_mask, wjets_mask, 
                                delta_r_threshold, met_threshold,
                                delta_list_met):

    # Muones
    good_muons_up = (muons_mask["up"]) & (
        delta_r_mask(events.Muon, electrons, threshold=delta_r_threshold)
    )
    muons_up = events.Muon[good_muons_up]


    good_muons_down = (muons_mask["down"]) & (
        delta_r_mask(events.Muon, electrons, threshold=delta_r_threshold)
    )
    muons_down = events.Muon[good_muons_down]
        

    # Taus
    good_taus_up = (
        (taus_mask["up"])
        & (delta_r_mask(events.Tau, electrons, threshold=delta_r_threshold))
        & (delta_r_mask(events.Tau, muons, threshold=delta_r_threshold))
    )
    taus_up = events.Tau[good_taus_up]

    good_taus_down = (
        (taus_mask["down"])
        & (delta_r_mask(events.Tau, electrons, threshold=delta_r_threshold))
        & (delta_r_mask(events.Tau, muons, threshold=delta_r_threshold))
    )
    taus_down = events.Tau[good_taus_down]


    # Bjets
    good_bjets_jes_up = (
        bjets_mask["JES_up"]
        & (delta_r_mask(jets_veto, electrons, threshold=delta_r_threshold))
        & (delta_r_mask(jets_veto, muons, threshold=delta_r_threshold))
        & (delta_r_mask(jets_veto, taus, threshold=delta_r_threshold))
    ) 
    bjets_jes_up = jets_veto[good_bjets_jes_up]


    good_bjets_jes_down = (
        bjets_mask["JES_down"]
        & (delta_r_mask(jets_veto, electrons, threshold=delta_r_threshold))
        & (delta_r_mask(jets_veto, muons, threshold=delta_r_threshold))
        & (delta_r_mask(jets_veto, taus, threshold=delta_r_threshold))
    )
    bjets_jes_down = jets_veto[good_bjets_jes_down]


    good_bjets_jer_up = (
        bjets_mask["JER_up"]
        & (delta_r_mask(jets_veto, electrons, threshold=delta_r_threshold))
        & (delta_r_mask(jets_veto, muons, threshold=delta_r_threshold))
        & (delta_r_mask(jets_veto, taus, threshold=delta_r_threshold))
    )
    bjets_jer_up = jets_veto[good_bjets_jer_up]


    good_bjets_jer_down = (
        bjets_mask["JER_down"]
        & (delta_r_mask(jets_veto, electrons, threshold=delta_r_threshold))
        & (delta_r_mask(jets_veto, muons, threshold=delta_r_threshold))
        & (delta_r_mask(jets_veto, taus, threshold=delta_r_threshold))
    )
    bjets_jer_down = jets_veto[good_bjets_jer_down]


    # light_jets
    good_jets_jes_up = (
        light_jets_mask["JES_up"]
        & (delta_r_mask(jets_veto, electrons, threshold=delta_r_threshold))
        & (delta_r_mask(jets_veto, muons, threshold=delta_r_threshold))
        & (delta_r_mask(jets_veto, taus, threshold=delta_r_threshold))
    )
    jets_jes_up = jets_veto[good_jets_jes_up]


    good_jets_jes_down = (
        light_jets_mask["JES_down"]
        & (delta_r_mask(jets_veto, electrons, threshold=delta_r_threshold))
        & (delta_r_mask(jets_veto, muons, threshold=delta_r_threshold))
        & (delta_r_mask(jets_veto, taus, threshold=delta_r_threshold))
    )
    jets_jes_down = jets_veto[good_jets_jes_down]


    good_jets_jer_up = (
        light_jets_mask["JER_up"]
        & (delta_r_mask(jets_veto, electrons, threshold=delta_r_threshold))
        & (delta_r_mask(jets_veto, muons, threshold=delta_r_threshold))
        & (delta_r_mask(jets_veto, taus, threshold=delta_r_threshold))
    )
    jets_jer_up = jets_veto[good_jets_jer_up]


    good_jets_jer_down = (
        light_jets_mask["JER_down"]
        & (delta_r_mask(jets_veto, electrons, threshold=delta_r_threshold))
        & (delta_r_mask(jets_veto, muons, threshold=delta_r_threshold))
        & (delta_r_mask(jets_veto, taus, threshold=delta_r_threshold))
    )
    jets_jer_down = jets_veto[good_jets_jer_down]


    # Fatjets
    good_fatjets_jes_up = (
        fatjets_mask["JES_up"]
        & (delta_r_mask(events.FatJet, electrons, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, muons, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, taus, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, bjets, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, jets, threshold = 2*delta_r_threshold))
    )   
    fatjets_jes_up = events.FatJet[good_fatjets_jes_up]


    good_fatjets_jes_down = (
        fatjets_mask["JES_down"]
        & (delta_r_mask(events.FatJet, electrons, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, muons, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, taus, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, bjets, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, jets, threshold = 2*delta_r_threshold))
    )   
    fatjets_jes_down = events.FatJet[good_fatjets_jes_down]


    good_fatjets_jer_up = (
        fatjets_mask["JER_up"]
        & (delta_r_mask(events.FatJet, electrons, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, muons, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, taus, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, bjets, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, jets, threshold = 2*delta_r_threshold))
    )   
    fatjets_jer_up = events.FatJet[good_fatjets_jer_up]


    good_fatjets_jer_down = (
        fatjets_mask["JER_down"]
        & (delta_r_mask(events.FatJet, electrons, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, muons, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, taus, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, bjets, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, jets, threshold = 2*delta_r_threshold))
    )   
    fatjets_jer_down = events.FatJet[good_fatjets_jer_down]


    # Wjets
    good_wjets_jes_up = (
        wjets_mask["JES_up"]
        & (delta_r_mask(events.FatJet, electrons, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, muons, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, taus, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, bjets, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, jets, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, fatjets, threshold = 2*delta_r_threshold))
    )   
    wjets_jes_up = events.FatJet[good_wjets_jes_up]


    good_wjets_jes_down = (
        wjets_mask["JES_down"]
        & (delta_r_mask(events.FatJet, electrons, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, muons, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, taus, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, bjets, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, jets, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, fatjets, threshold = 2*delta_r_threshold))
    )   
    wjets_jes_down = events.FatJet[good_wjets_jes_down] 


    good_wjets_jer_up = (
        wjets_mask["JER_up"]
        & (delta_r_mask(events.FatJet, electrons, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, muons, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, taus, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, bjets, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, jets, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, fatjets, threshold = 2*delta_r_threshold))
    )  
    wjets_jer_up = events.FatJet[good_wjets_jer_up]


    good_wjets_jer_down = (
        wjets_mask["JER_down"]
        & (delta_r_mask(events.FatJet, electrons, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, muons, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, taus, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, bjets, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, jets, threshold = 2*delta_r_threshold))
        & (delta_r_mask(events.FatJet, fatjets, threshold = 2*delta_r_threshold))
    )   
    wjets_jer_down = events.FatJet[good_wjets_jer_down]



    map_met_mask = one_var_per_event_MET(met_threshold, delta_list_met, events.MET)


    # ------------------------------------------
    # Fatjets are not defined in all the samples
    # ------------------------------------------
    has_fatjets = np.sum(ak.num(events.FatJet)) > 0

    def met_exists(mask_map, key):
        return key in mask_map and "up" in mask_map[key] and "down" in mask_map[key]


    output_map = {
        "ROCHESTER": {
            "up": {
                "one_muon": (ak.num(muons_up) == 1),
                "muon_veto": (ak.num(muons_up) == 0),
                "muon": muons_up,
                "new_met_pt": map_met_mask["Muon"]["up"]["new_met_pt"],
                "new_met_phi": map_met_mask["Muon"]["up"]["new_met_phi"],
                f"met_{met_threshold}": map_met_mask["Muon"]["up"]["mask"]
            },
            "down": {
                "one_muon": (ak.num(muons_down) == 1),
                "muon_veto": (ak.num(muons_down) == 0),
                "muon": muons_down,
                "new_met_pt": map_met_mask["Muon"]["down"]["new_met_pt"],
                "new_met_phi": map_met_mask["Muon"]["down"]["new_met_phi"],
                f"met_{met_threshold}": map_met_mask["Muon"]["down"]["mask"]
            }
        },
        "TES": {
            "up": {
                "one_tau": (ak.num(taus_up) == 1),
                "tau_veto": (ak.num(taus_up) == 0),
                "tau": taus_up,
                "new_met_pt": map_met_mask["Tau"]["up"]["new_met_pt"],
                "new_met_phi": map_met_mask["Tau"]["up"]["new_met_phi"],
                f"met_{met_threshold}": map_met_mask["Tau"]["up"]["mask"]
            },
            "down": {
                "one_tau": (ak.num(taus_down) == 1),
                "tau_veto": (ak.num(taus_down) == 0),
                "tau": taus_down,
                "new_met_pt": map_met_mask["Tau"]["down"]["new_met_pt"],
                "new_met_phi": map_met_mask["Tau"]["down"]["new_met_phi"],
                f"met_{met_threshold}": map_met_mask["Tau"]["down"]["mask"]
            }
        },
        "jet_JES": {
            "up": {
                "one_bjet": (ak.num(bjets_jes_up) == 1),
                "bjet_veto": (ak.num(bjets_jes_up) == 0),
                "bjet": bjets_jes_up,
                "jet": jets_jes_up,
                "new_met_pt": map_met_mask["Jet_JES"]["up"]["new_met_pt"],
                "new_met_phi": map_met_mask["Jet_JES"]["up"]["new_met_phi"],
                f"met_{met_threshold}": map_met_mask["Jet_JES"]["up"]["mask"]
            },
            "down": {
                "one_bjet": (ak.num(bjets_jes_down) == 1),
                "bjet_veto": (ak.num(bjets_jes_down) == 0),
                "bjet": bjets_jes_down,
                "jet": jets_jes_down,
                "new_met_pt": map_met_mask["Jet_JES"]["down"]["new_met_pt"],
                "new_met_phi": map_met_mask["Jet_JES"]["down"]["new_met_phi"],
                f"met_{met_threshold}": map_met_mask["Jet_JES"]["down"]["mask"]
            }
        },
        "jet_JER": {
            "up": {
                "one_bjet": (ak.num(bjets_jer_up) == 1),
                "bjet_veto": (ak.num(bjets_jer_up) == 0),
                "bjet": bjets_jer_up,
                "jet": jets_jer_up,
                "new_met_pt": map_met_mask["Jet_JER"]["up"]["new_met_pt"],
                "new_met_phi": map_met_mask["Jet_JER"]["up"]["new_met_phi"],
                f"met_{met_threshold}": map_met_mask["Jet_JER"]["up"]["mask"]
            },
            "down": {
                "one_bjet": (ak.num(bjets_jer_down) == 1),
                "bjet_veto": (ak.num(bjets_jer_down) == 0),
                "bjet": bjets_jer_down,
                "jet": jets_jer_down,
                "new_met_pt": map_met_mask["Jet_JER"]["down"]["new_met_pt"],
                "new_met_phi": map_met_mask["Jet_JER"]["down"]["new_met_phi"],
                f"met_{met_threshold}": map_met_mask["Jet_JER"]["down"]["mask"]
            }
        },
        "met_UNCLUSTERED": {
            "up": {
                "new_met_pt": map_met_mask["MET_uncluster"]["up"]["new_met_pt"],
                "new_met_phi": map_met_mask["MET_uncluster"]["up"]["new_met_phi"],
                f"met_{met_threshold}": map_met_mask["MET_uncluster"]["up"]["mask"]
            },
            "down": {
                "new_met_pt": map_met_mask["MET_uncluster"]["down"]["new_met_pt"],
                "new_met_phi": map_met_mask["MET_uncluster"]["down"]["new_met_phi"],
                f"met_{met_threshold}": map_met_mask["MET_uncluster"]["down"]["mask"]
            }
        }
    }

    if has_fatjets and met_exists(map_met_mask, "FatJet_JES") and met_exists(map_met_mask, "FatJet_JER"):
        output_map["fatjet_JES"] = {
            "up": {
                "fatjet": fatjets_jes_up,
                "wjet": wjets_jes_up,
                "new_met_pt": map_met_mask["FatJet_JES"]["up"]["new_met_pt"],
                "new_met_phi": map_met_mask["FatJet_JES"]["up"]["new_met_phi"],
                f"met_{met_threshold}": map_met_mask["FatJet_JES"]["up"]["mask"]
            },
            "down": {
                "fatjet": fatjets_jes_down,
                "wjet": wjets_jes_down,
                "new_met_pt": map_met_mask["FatJet_JES"]["down"]["new_met_pt"],
                "new_met_phi": map_met_mask["FatJet_JES"]["down"]["new_met_phi"],
                f"met_{met_threshold}": map_met_mask["FatJet_JES"]["down"]["mask"]
            }
        }
    
        output_map["fatjet_JER"] = {
            "up": {
                "fatjet": fatjets_jer_up,
                "wjet": wjets_jer_up,
                "new_met_pt": map_met_mask["FatJet_JER"]["up"]["new_met_pt"],
                "new_met_phi": map_met_mask["FatJet_JER"]["up"]["new_met_phi"],
                f"met_{met_threshold}": map_met_mask["FatJet_JER"]["up"]["mask"]
            },
            "down": {
                "fatjet": fatjets_jer_down,
                "wjet": wjets_jer_down,
                "new_met_pt": map_met_mask["FatJet_JER"]["down"]["new_met_pt"],
                "new_met_phi": map_met_mask["FatJet_JER"]["down"]["new_met_phi"],
                f"met_{met_threshold}": map_met_mask["FatJet_JER"]["down"]["mask"]
            }
        }

    """
    output_map = {
    "ROCHESTER": {
        "up": {
            "one_muon": (ak.num(muons_up) == 1),
            "muon_veto": (ak.num(muons_up) == 0),
            "muon": muons_up, 
            "new_met_pt": map_met_mask["Muon"]["up"]["new_met_pt"],
            "new_met_phi": map_met_mask["Muon"]["up"]["new_met_phi"],
            f"met_{met_threshold}": map_met_mask["Muon"]["up"]["mask"]
        },
        "down": {
            "one_muon": (ak.num(muons_down) == 1),
            "muon_veto": (ak.num(muons_down) == 0),
            "muon": muons_down,
            "new_met_pt": map_met_mask["Muon"]["down"]["new_met_pt"],
            "new_met_phi": map_met_mask["Muon"]["down"]["new_met_phi"],
            f"met_{met_threshold}": map_met_mask["Muon"]["down"]["mask"]               
        }
    },
    "TES": {
        "up": {
            "one_tau": (ak.num(taus_up) == 1),
            "tau_veto": (ak.num(taus_up) == 0),
            "tau": taus_up,
            "new_met_pt": map_met_mask["Tau"]["up"]["new_met_pt"],
            "new_met_phi": map_met_mask["Tau"]["up"]["new_met_phi"],
            f"met_{met_threshold}": map_met_mask["Tau"]["up"]["mask"]
        },
        "down": {
            "one_tau": (ak.num(taus_down) == 1),
            "tau_veto": (ak.num(taus_down) == 0),  
            "tau": taus_down,
            "new_met_pt": map_met_mask["Tau"]["down"]["new_met_pt"],
            "new_met_phi": map_met_mask["Tau"]["down"]["new_met_phi"],
            f"met_{met_threshold}": map_met_mask["Tau"]["down"]["mask"]           
        }
    },
    "jet_JES": {
        "up": {
            "one_bjet": (ak.num(bjets_jes_up) == 1),
            "bjet_veto": (ak.num(bjets_jes_up) == 0),
            # "one_jet": (ak.num(jets_jes_up) == 1),
            # "veto_jet": (ak.num(jets_jes_up) == 0),
            "bjet": bjets_jes_up,
            "jet": jets_jes_up,
            "new_met_pt": map_met_mask["Jet_JES"]["up"]["new_met_pt"],
            "new_met_phi": map_met_mask["Jet_JES"]["up"]["new_met_phi"],
            f"met_{met_threshold}": map_met_mask["Jet_JES"]["up"]["mask"]
        },
        "down": {
            "one_bjet": (ak.num(bjets_jes_down) == 1),
            "bjet_veto": (ak.num(bjets_jes_down) == 0),
            # "one_jet": (ak.num(jets_jes_down) == 1),
            # "veto_jet": (ak.num(jets_jes_down) == 0),
            "bjet": bjets_jes_down,
            "jet": jets_jes_down,
            "new_met_pt": map_met_mask["Jet_JES"]["down"]["new_met_pt"],
            "new_met_phi": map_met_mask["Jet_JES"]["down"]["new_met_phi"],
            f"met_{met_threshold}": map_met_mask["Jet_JES"]["down"]["mask"]
        },
    },
    "jet_JER":{ 
        "up": {
            "one_bjet": (ak.num(bjets_jer_up) == 1),
            "bjet_veto": (ak.num(bjets_jer_up) == 0),
            # "one_jet": (ak.num(jets_jer_up) == 1),
            # "jet_veto": (ak.num(jets_jer_up) == 0),
            "bjet": bjets_jer_up,
            "jet": jets_jer_up,
            "new_met_pt": map_met_mask["Jet_JER"]["up"]["new_met_pt"],
            "new_met_phi": map_met_mask["Jet_JER"]["up"]["new_met_phi"],
            f"met_{met_threshold}": map_met_mask["Jet_JER"]["up"]["mask"]
        },
        "down": {
            "one_bjet": (ak.num(bjets_jer_down) == 1),
            "bjet_veto": (ak.num(bjets_jer_down) == 0),
            # "one_jet": (ak.num(jets_jer_down) == 1),
            # "jet_veto": (ak.num(jets_jer_down) == 0),
            "bjet": bjets_jer_down,
            "jet": jets_jer_down,
            "new_met_pt": map_met_mask["Jet_JER"]["down"]["new_met_pt"],
            "new_met_phi": map_met_mask["Jet_JER"]["down"]["new_met_phi"],
            f"met_{met_threshold}": map_met_mask["Jet_JER"]["down"]["mask"]
        }
    },
    "fatjet_JES": {
        "up": {
            # "one_fatjet": (ak.num(fatjets_jes_up) == 1),
            # "veto_fatjet": (ak.num(fatjets_jes_up) == 0),
            # "one_wjet": (ak.num(wjets_jes_up) == 1),
            # "veto_wjet": (ak.num(wjets_jes_up) == 0),
            "fatjet": fatjets_jes_up,
            "wjet": wjets_jes_up,
            "new_met_pt": map_met_mask["FatJet_JES"]["up"]["new_met_pt"],
            "new_met_phi": map_met_mask["FatJet_JES"]["up"]["new_met_phi"],
            f"met_{met_threshold}": map_met_mask["FatJet_JES"]["up"]["mask"]
        },
        "down": {
            # "one_fatjet": (ak.num(fatjets_jes_down) == 1),
            # "veto_fatjet": (ak.num(fatjets_jes_down) == 0),
            # "one_wjet": (ak.num(wjets_jes_down) == 1),
            # "veto_wjet": (ak.num(wjets_jes_down) == 0),
            "fatjet": fatjets_jes_down,
            "wjet": wjets_jes_down,
            "new_met_pt": map_met_mask["FatJet_JES"]["down"]["new_met_pt"],
            "new_met_phi": map_met_mask["FatJet_JES"]["down"]["new_met_phi"],
            f"met_{met_threshold}": map_met_mask["FatJet_JES"]["down"]["mask"]               
        }

    },
    "fatjet_JER": {
        "up": {
            # "one_fatjet": (ak.num(fatjets_jer_up) == 1),
            # "veto_fatjet": (ak.num(fatjets_jer_up) == 0),
            # "one_wjet": (ak.num(wjets_jer_up) == 1),
            # "veto_wjet": (ak.num(wjets_jer_up) == 0),
            "fatjet": fatjets_jer_up,
            "wjet": wjets_jer_up,
            "new_met_pt": map_met_mask["FatJet_JER"]["up"]["new_met_pt"],
            "new_met_phi": map_met_mask["FatJet_JER"]["up"]["new_met_phi"],
            f"met_{met_threshold}": map_met_mask["FatJet_JER"]["up"]["mask"]
        },
        "down": {
            # "one_fatjet": (ak.num(fatjets_jer_down) == 1),
            # "veto_fatjet": (ak.num(fatjets_jer_down) == 0),
            # "one_wjet": (ak.num(wjets_jer_up) == 1),
            # "veto_wjet": (ak.num(wjets_jer_up) == 0),
            "fatjet": fatjets_jer_down,
            "wjet": wjets_jer_down,
            "new_met_pt": map_met_mask["FatJet_JER"]["down"]["new_met_pt"],
            "new_met_phi": map_met_mask["FatJet_JER"]["down"]["new_met_phi"],
            f"met_{met_threshold}": map_met_mask["FatJet_JER"]["down"]["mask"]               
        }
    },
    "met_UNCLUSTERED": {
        "up": {
            "new_met_pt": map_met_mask["MET_uncluster"]["up"]["new_met_pt"],
            "new_met_phi": map_met_mask["MET_uncluster"]["up"]["new_met_phi"],
            f"met_{met_threshold}": map_met_mask["MET_uncluster"]["up"]["mask"]
        },
        "down": {
            "new_met_pt": map_met_mask["MET_uncluster"]["down"]["new_met_pt"],
            "new_met_phi": map_met_mask["MET_uncluster"]["down"]["new_met_phi"],
            f"met_{met_threshold}": map_met_mask["MET_uncluster"]["down"]["mask"]
        }
    }
    }
    """


    return output_map
    











 