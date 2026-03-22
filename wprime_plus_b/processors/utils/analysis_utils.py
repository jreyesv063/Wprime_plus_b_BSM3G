import os
import re
import json
import numpy as np
import pandas as pd
import awkward as ak
import pyarrow as pa
import importlib.resources
import pyarrow.parquet as pq
from datetime import datetime
from typing import List, Union
from coffea.nanoevents.methods import candidate, vector

from concurrent.futures import ThreadPoolExecutor, as_completed



def delta_r_mask(first: ak.Array, second: ak.Array, threshold: float) -> ak.Array:
    """
    Select objects from 'first' which are at least threshold away from all objects in 'second'.
    The result is a mask (i.e., a boolean array) of the same shape as first.
    
    Parameters:
    -----------
    first: 
        objects which are required to be at least threshold away from all objects in second
    second: 
        objects which are all objects in first must be at leats threshold away from
    threshold: 
        minimum delta R between objects

    Return:
    -------
        boolean array of objects in objects1 which pass delta_R requirement
    """
    mval = first.metric_table(second)
    return ak.all(mval > threshold, axis=-1)



def cross_cleaning(objects, cc):
    """
    Perform cross-cleaning between reconstructed physics objects based on
    their angular separation (ΔR).

    The cleaning strategy follows these principles:
      - Leptons have priority over jets and fatjets.
      - Small-radius jets (AK4) have priority over fatjets (AK8). They are acleaned against leptons.
      - Fatjets are cleaned against leptons and small jets, but not among themselves.
      - topjets and wjets are not cross-cleaned with each other, as they belong to different and mutually exclusive reconstruction cases (top tagger).

    * Object priority (highest → lowest):
                leptons → small-radius jets (AK4) → fatjets (AK8)      
    """

    # Objects that should not be cross-cleaned
    skip_keys = ["met", "events", "pdf_nominal", "pdf_ratio"]

    # If flavored jets exist, do not clean the inclusive jet collection, since jets already contain bjets, cjets, and lightjets
    if any(k in objects for k in ("bjets", "cjets", "lightjets")):
        skip_keys.append("jets")

    #  # Objects participating in cross-cleaning
    clean_objects = {
        k: v for k, v in objects.items() if k not in skip_keys
    }

    cleaned_objects = {}
    leptons = {"electrons", "muons", "taus"}
    small_jets = {"jets", "bjets", "lightjets"}   # AK4 jets
    fatjets = {"topjets", "wjets"}                # AK8 jets

    for name_i, obj_i in clean_objects.items():

        # Use a larger ΔR for fatjets
        cc_i = 2 * cc if name_i in fatjets else cc 
        
        mask = ak.ones_like(obj_i.pt, dtype=bool)
        
        for name_j, obj_j in clean_objects.items():
            if name_i == name_j:
                # Do not clean an object collection against itself
                continue

            # ─────────────────────────────────────────────
            # Leptons:
            #   - Clean against other leptons
            #   - Do NOT clean against jets or fatjets
            if name_i in leptons:
                if name_j in fatjets or name_j in small_jets:
                    continue
                    
            # ─────────────────────────────────────────────
            # Small-radius jets (AK4):
            #   - Clean against leptons and other small jets
            #   - Do NOT clean against fatjets (AK8)
            if name_i in small_jets:
                if name_j in fatjets:
                    continue
                elif name_j in small_jets:
                    continue
        
            # ─────────────────────────────────────────────
            # Fatjets (AK8):
            #   - Clean against leptons and small jets
            #   - Do NOT clean among themselves (no topjets ↔ wjets)
            if name_i in fatjets:
                if name_j in fatjets:
                    continue

            # Apply ΔR-based cleaning
            mask = mask & delta_r_mask(obj_i, obj_j, threshold=cc_i)
    
            cleaned_objects[name_i] = obj_i[mask]

    # Add back objects that were excluded from cleaning
    for key in skip_keys:
        if key in objects:
            cleaned_objects[key] = objects[key]

    return cleaned_objects
    
    
def apply_selection(objects, mask):
    """
    Apply an event-level boolean mask to all objects in a dictionary.

    Parameters
    ----------
    objects : dict
        Dictionary containing awkward arrays of physics objects
        (e.g. jets, bjets, electrons, muons, MET, etc.).
    mask : awkward.Array (bool)
        Boolean mask at event level selecting the events of interest.

    Returns
    -------
    dict
        New dictionary where each object has been filtered
        according to the provided event mask.
    """
    return {key: obj[mask] for key, obj in objects.items()}
    

    
def get_mask_until_object(selections, cuts, obj_name, include_cut=True, only_cut = False):
    """
    Parameters
    ----------
    selections : PackedSelection (or equivalent wrapper)
        Object containing all individual cut masks.

    cuts : list[str]
        Ordered list of cut names defining the region cutflow.

    obj_name : str
        Physics object name (e.g. 'tau', 'muon', 'electron').

    Returns
    -------
    awkward.Array
        Cumulative mask up to the matched cut.
        If no match is found, returns a False mask
        with the correct event size.
    """

    # Build a simple plural form automatically
    plural = obj_name + "s"

    # Define internally all patterns that should trigger
    # object-dependent corrections
    patterns = [
        f"one_{obj_name}",
        f"two_{plural}",
        f"at_least_one_{obj_name}",
        f"at_least_two_{plural}",
        f"{obj_name}_veto",
    ]

    # Loop over the cutflow in order (respect analysis logic)
    for i, cut in enumerate(cuts):

        # Check if the current cut matches any of the object patterns
        for pattern in patterns:
            #if re.fullmatch(pattern, cut):
            if pattern in cut:   
                # returns the complete mask of the cut.
                if only_cut:
                    return selections.all(cut)
                

                # Return the matched cut name and the cumulative mask
                # including all cuts up to this point
                idx = i + 1 if include_cut else i
                
                return selections.all(*cuts[:idx])


    if obj_name == "top_tagger":
        for i, cut in enumerate(cuts):
            if "top_tagger" in cut:

                
                # returns the complete mask of the cut.
                if only_cut:
                    return selections.all(cut)

                # Return the matched cut name and the cumulative mask
                # including all cuts up to this point
                idx = i + 1 if include_cut else i
                
                return selections.all(*cuts[:idx])

    
    if obj_name == "trigger":
        for i, cut in enumerate(cuts):
            if cut ==  "trigger":

                # returns the complete mask of the cut.
                if only_cut:
                    return selections.all(cut)

                # Return the matched cut name and the cumulative mask
                # including all cuts up to this point
                idx = i + 1 if include_cut else i
                return selections.all(*cuts[:idx])
         
    return np.zeros_like(selections.all(cuts[0]), dtype=bool)


# ===================================================
#  Metadata
# ===================================================

def fill_cutflow(cut_names, selections, table_name, metadata, weights):
        
    # Weighted events
    metadata.update({table_name: {}})
    # Raw events
    metadata.update({f"{table_name}_raw": {}})
    

    # ==========================
    #   Initial values
    # ==========================
    metadata[table_name]["sumw"] = ak.sum(weights)
    metadata[f"{table_name}_raw"]["sumw"] = len(weights)
    
    
    cuts_applied = [] 
    for cut_name in cut_names:
        cuts_applied.append(cut_name)
        current_selection = selections.all(*cuts_applied)
        metadata[table_name][cut_name] = ak.sum(weights[current_selection])
        metadata[f"{table_name}_raw"][cut_name] = len(weights[current_selection])
        
    #metadata.update({"sumw": ak.sum(weights)})



# ===================================================
#  Parallelization
# ===================================================
def parallel_processing(key, func):
    """
    Executes a function (top tagger scenario) using a key (case).

    Returns
    -------
    tuple
        (key, tops_array, mask_array)
    """    
    # Execute the scenario: returns tops found and a boolean mask
    t, p,  m = func()    

    return key, t, p,  m
    
    

# =====================================================
#      Top tagger
# ======================================================
def pdg_masses():
    
    with open("wprime_plus_b/json_files/wAndtop_masses.json", "r") as f:
        pdg = json.load(f)
                    
    top_mass_pdg = pdg['pdg']['top_mass']           
    w_mass_pdg = pdg['pdg']['w_mass'] 
    
    return top_mass_pdg, w_mass_pdg


def tagger_constants(case: str = "hadronic"):
    # W, top and chi2
    with open("wprime_plus_b/json_files/wAndtop_masses.json", "r") as f:
        masses = json.load(f)


    top_sigma = masses[case]['top_sigma']
    top_low_mass = masses[case]['top_low_mass']
    top_up_mass = masses[case]['top_up_mass']


    w_sigma = masses[case]['w_sigma']
    w_low_mass = masses[case]['w_low_mass']
    w_up_mass = masses[case]['w_up_mass']  

    chi2 = masses[case]['chi2']

    return top_sigma, top_low_mass, top_up_mass, w_sigma, w_low_mass, w_up_mass, chi2


def chi2_test(topJet, wJet, top_sigma, w_sigma, top_mass_pdg,  w_mass_pdg):
        
    t = (topJet.mass - top_mass_pdg) / top_sigma
    w = (wJet.mass - w_mass_pdg) / w_sigma
    

    chi2 = t**2 + w**2
    

    return chi2

# =================================================
#  Systematic variations
# =================================================
# Function that decides whether a cut applies to the object
def check_object_cut_dependency(obj, cut):
    # Direct match with the object
    if obj in cut:
        return True
    
    # Changes in leptons affect met and delta_phi_jet_met
    if obj in {"electron", "muon", "tau"} and cut in {"met", "delta_phi_jet_met"}:
        return True

    # Changes in AK4/AK8 Jets affect met and delta_phi_jet_met, also the top_tagger
    if obj in {"bjet", "cjet", "lightjet", "topjet", "wjet"}:
        if cut in {"met", "delta_phi_jet_met"}:
            return True
        if "top_tagger" in cut:
            return True

    # Met applies only to met and delta_phi_jet_met
    if obj == "met" and cut in {"met", "delta_phi_jet_met"}:
        return True

    return False
    

def map_object_level_var(case: str = "AK4_JES"):

    map_names = {
        "AK4_JES": "CMS_scale_j",
        "AK4_JER": "CMS_res_j",
        "AK8_JES": "CMS_scale_fj",
        "AK8_JER": "CMS_res_fj",
        "lepton_Rochester": "CMS_scale_m",
        "lepton_TES": "CMS_scale_t",
        "lepton_SS": "CMS_scale_e_13TeV",
        "met_Uncluster": "CMS_scale_met_unclustered_energy",
    }

    return map_names[case]
