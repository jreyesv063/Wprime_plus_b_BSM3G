import json
import numpy as np
import awkward as ak
import importlib.resources


def get_trigger_mask(
    events: ak.Array,
    lepton_flavor: str,
    year: str,
    trigger_case: str,
    Or_HLT: bool
):
  
    if lepton_flavor not in ["mu", "ele", "tau", "ditau", "muon_highPt"]:
        raise ValueError(f"Unknown lepton flavor '{lepton_flavor}'")


    # Load triggers from JSON
    with open("wprime_plus_b/json_files/triggers.json", "r") as f:
        trigger_name = json.load(f)["HLT_names"][year][trigger_case] 
        
    if Or_HLT:
        all_triggers = [
            trig for trig in events.HLT.fields if any(trig.startswith(r) for r in trigger_name)
        ]
    else:
        all_triggers = trigger_name

    # Combine masks for all reference triggers 
    masks = [events.HLT[trig] for trig in all_triggers if trig in events.HLT.fields]        
        
    trigger_mask = ak.any(ak.Array(masks), axis=0)


    return all_triggers, trigger_mask

    

def get_trigger_match_mask(
    objects: dict,
    year: str,    
    lepton_flavor: str,
    trigger_names: list 
):

    """
    TrigObj is a collection of trigger objects (electrons, muons, taus, jets, MET, etc.) that passed some HLT filter in the event. It is used to:
    - trigger matching
    - efficiency studies
    - path validation

    ** TrigObj_id: See triggers.json
    ** Kinematics variables: used for matching with offline objects:
        - pt
        - eta
        - phi
    ** Each bit indicates that the object passed a specific HLT filter.

    filterBits = 34 = 32 + 2 -> bits 1 and 5 are actived.

    Ref: https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgammaNanoAOD#Trigger_bits_how_to
     
     np.unique(ak.flatten(events.TrigObj.id))
    
    """
    run = "run2" if year in ["2016APV", "2016", "2017", "2018"] else "run3"
    base_name = list({name.split("_")[0] for name in trigger_names})

    events = objects["events"]
    
    lepton_map = {
        "mu": objects["muons"],
        "ele": objects["electrons"],
        "tau": objects["taus"],
    }
    
    leptons = lepton_map[lepton_flavor]
    
    # Load triggers from JSON
    with open("wprime_plus_b/json_files/triggers.json", "r") as f:
        trigger_match_fields = json.load(f)[f"trigger_match_{run}"]
    
    
    # ====================================================
    #  Mask
    # ====================================================
    trigger_match_mask = np.zeros(len(events), dtype=bool)
    
    for name in base_name:
        if name not in trigger_match_fields:
            return ~trigger_match_mask


        cfg = trigger_match_fields[name]
        trigobjs = events.TrigObj

        trigobj_mask = (
            (abs(trigobjs.id) == cfg["id"])
            & (trigobjs.pt >= cfg["pt"])
            & ((trigobjs.filterBits & cfg["filterBits"]) > 0)
        )

        selected_trigobjs = trigobjs[trigobj_mask]

        # --- ΔR matching ---
        delta_r = leptons.metric_table(selected_trigobjs)
        matched_per_lepton = ak.any(delta_r < cfg["delta_r"], axis=2)
        matched = ak.any(matched_per_lepton, axis=1)
        
        trigger_match_mask = trigger_match_mask | matched

    return trigger_match_mask