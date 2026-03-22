import gzip
import json
import cloudpickle
import numpy as np
import awkward as ak
from typing import Tuple
from pathlib import Path
import importlib.resources
from coffea.lookup_tools import extractor
from coffea.nanoevents.methods.base import NanoEventsArray

from coffea.nanoevents.methods import vector

# Local library
from wprime_plus_b.corrections.met import update_met

# External libraries
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory, CorrectedMETFactory


# JER: https://github.com/cms-jet/JRDatabase/tree/master/textFiles
# JEC: https://github.com/cms-jet/JECDatabase/tree/master/textFiles

# ===================================================================
#    Utils
# ===================================================================
def add_jec_variables(jets: ak.Array, event_rho: ak.Array):
    jets["pt_raw"] = (1 - jets.rawFactor) * jets.pt
    jets["mass_raw"] = (1 - jets.rawFactor) * jets.mass
    jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]

    if hasattr(jets, "matched_gen"):
        # Only MC samples
        jets["pt_gen"] = ak.values_astype(
            ak.fill_none(jets.matched_gen.pt, 0), np.float32
        )
    return jets


def get_jet_factory(files):
    base_path = Path.cwd() / "wprime_plus_b" / "data"
    jec_name_map = {
        "JetPt": "pt",
        "JetMass": "mass",
        "JetEta": "eta",
        "JetA": "area",
        "ptGenJet": "pt_gen",
        "ptRaw": "pt_raw",
        "massRaw": "mass_raw",
        "Rho": "event_rho",
        "METpt": "pt",
        "METphi": "phi",
        "JetPhi": "phi",
        "UnClusteredEnergyDeltaX": "MetUnclustEnUpDeltaX",
        "UnClusteredEnergyDeltaY": "MetUnclustEnUpDeltaY",
    }
    ext = extractor()
    ext.add_weight_sets([f"* * {Path(base_path, file)}" for file in files])
    ext.finalize()
    jec_stack = JECStack(ext.make_evaluator())
    
    return CorrectedJetsFactory(jec_name_map, jec_stack)

def get_met_factory():
    jec_name_map = {
        "JetPt": "pt",
        "JetMass": "mass",
        "JetEta": "eta",
        "JetA": "area",
        "ptGenJet": "pt_gen",
        "ptRaw": "pt_raw",
        "massRaw": "mass_raw",
        "Rho": "event_rho",
        "METpt": "pt",
        "METphi": "phi",
        "JetPhi": "phi",
        "UnClusteredEnergyDeltaX": "MetUnclustEnUpDeltaX",
        "UnClusteredEnergyDeltaY": "MetUnclustEnUpDeltaY",
    }
    
    return CorrectedMETFactory(jec_name_map)

    

def apply_jet_corrections(events: NanoEventsArray, year: str, syst_var: bool, jet_case: str):
    jet_type = "Jet" if jet_case == "AK4" else "FatJet"
    met_type = "MET" if year in ["2016APV", "2016", "2017", "2018"] else "PuppiMET"
    
    # ============================================
    #  Load corrections
    # ============================================    
    # Correction name
    with open(f"wprime_plus_b/corrections/correction_names/{jet_case}.json", "r") as f:
        names = json.load(f)[year]


    # ============================================
    # Create jec factories
    # ============================================
    sample_type = "MC" if hasattr(events, "genWeight") else "DATA"


    if sample_type == "DATA":
        era_name = None
    
        # Load run number ranges
        with open(f"wprime_plus_b/corrections/correction_names/run_numbers.json", "r") as f:
            runs = json.load(f)[year] # Dictionary of eras with min/max run numbers
    
        run_min, run_max = np.min(events.run), np.max(events.run)

        era_name = next(
            (era["era"] for era in runs["eras"] 
             if run_min >= era["run_min"] and run_max <= era["run_max"]),
            None
        )

        if era_name is None:
            raise ValueError(f"No era found for run range {run_min}-{run_max}")


    files = []

    # -------
    # JEC
    # -------    
    submap = "nominal" if sample_type is "MC" else era_name
    if sample_type == "MC":
        base_path = f"JEC/MC/{year}"
    else:
        base_path = f"JEC/DATA/{year}/Run{era_name}"
    
    
    files.extend([
        f"{base_path}/{names[sample_type]['JEC'][submap]['L1FastJet']}.jec.txt",
        f"{base_path}/{names[sample_type]['JEC'][submap]['L2Relative']}.jec.txt",
        f"{base_path}/{names[sample_type]['JEC'][submap]['L3Absolute']}.jec.txt",
    ])

    if sample_type == "DATA":
        # ----------------
        # L2L3: Only DATA
        # ----------------
        files.append(
            f"{base_path}/{names[sample_type]['JEC'][submap]['L2L3Residual']}.jec.txt"
        )

    else:
        # ----------------
        # JER: Only MC
        # ----------------
        files.extend([
            # JER
            f"JER/MC/{year}/{names[sample_type]['JER']['ptResolution']}.jr.txt",
            f"JER/MC/{year}/{names[sample_type]['JER']['ScaleFactor']}.jersf.txt",
            # Uncertainties
            f"JEC/MC/{year}/{names[sample_type]['Uncertainties']['Sources']}.junc.txt",
            f"JEC/MC/{year}/{names[sample_type]['Uncertainties']['Total']}.junc.txt"
        ])

        

    #jet_factory = {}
    jet_factory = get_jet_factory(files)
    
            
    # Store original pt
    events[jet_type, "pt_nano"] =  events[jet_type].pt


    # Avoid samples without jets.
    if (np.sum(ak.num(events[jet_type])) != 0):
        # get corrected jets
        events[jet_type] = jet_factory.build(
            add_jec_variables(events[jet_type], events.fixedGridRhoFastjetAll),
            events.caches[0],
        )

    # ==============================================
    #  MET recalculation
    # ==============================================
    if jet_case == "AK4":
        events[met_type, "pt_nano"] = events[met_type, "pt"]
        events[met_type, "phi_nano"] = events[met_type, "phi"]

        met_factory = get_met_factory()
        events[met_type] = met_factory.build(events[met_type], events[jet_type], {})
        
    
    # =======================================
    # Remove unnecesary fields
    # =======================================
    keep_systematics = {
        "JES_pt_up",
        "JES_pt_down",
        "JER_pt_up",
        "JER_pt_down",
    }

    def drop_jet_field(name):
        if name.startswith(("JES", "JER")) and name not in keep_systematics:
            return False
        if name.startswith("jet_energy_"):
            return False
        if name in {"pt_jec", "mass_jec", "pt_jer", "mass_jer", "jet_resolution_rand_gauss"}:
            return False
        return True


    
    # ===============================================
    #  Systematic variations
    # ===============================================      
    if syst_var and sample_type == "MC":
        # Samples without Jets
        if (np.sum(ak.num(events[jet_type])) == 0):
            # Jet
            for var in ["JES_pt_up", "JES_pt_down", "JER_pt_up", "JER_pt_down"]:
                events[jet_type, var] = events[jet_type].pt
        
            # MET (no recalculation)
            for syst in [f"{jet_case}_JES_up", f"{jet_case}_JES_down", f"{jet_case}_JER_up", f"{jet_case}_JER_down"]:
                for var in ["pt", "phi"]:
                    events[met_type, f"{var}_{syst}"] = getattr(events.MET, var)
        
            return

        
        # JES
        events[jet_type, "JES_pt_up"] =  events[jet_type].JES_jes.up.pt
        events[jet_type, "JES_pt_down"] = events[jet_type].JES_jes.down.pt
        

        # JER        
        events[jet_type, "JER_pt_up"] =  events[jet_type].JER.up.pt       
        events[jet_type, "JER_pt_down"] =  events[jet_type].JER.down.pt
        

        # ============================================
        # MET variations in the Up/Down directions
        # ============================================
        if jet_case == "AK4":
            # JES
            events[met_type, f"pt_{jet_case}_JES_up"] = events[met_type].JES_jes.up.pt
            events[met_type, f"phi_{jet_case}_JES_up"] = events[met_type].JES_jes.up.phi
                
            events[met_type, f"pt_{jet_case}_JES_down"] = events[met_type].JES_jes.down.pt
            events[met_type, f"phi_{jet_case}_JES_down"] = events[met_type].JES_jes.down.phi

            # JER
            events[met_type, f"pt_{jet_case}_JER_up"] = events[met_type].JER.up.pt
            events[met_type, f"phi_{jet_case}_JER_up"] = events[met_type].JER.up.phi
    
            events[met_type, f"pt_{jet_case}_JER_down"] = events[met_type].JER.down.pt
            events[met_type, f"phi_{jet_case}_JER_down"] = events[met_type].JER.down.phi    

    
    # ===============================================
    # Remove extra fields from systematic variations
    # to free memory
    # ===============================================
    # JETs
    keep_jet = [f for f in events[jet_type].fields if drop_jet_field(f)]
    events[jet_type] = events[jet_type][keep_jet]
    events[jet_type] = ak.with_name(events[jet_type], "PtEtaPhiMLorentzVector")

    # MET
    keep = [
        f for f in events[met_type].fields
        if not (f.startswith("JES") or f.startswith("JER"))
    ]

    events[met_type] = events[met_type][keep]
    events[met_type] = ak.with_name(events[met_type], "PtEtaPhiMLorentzVector")

