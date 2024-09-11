import os
import json
import numpy as np
import pandas as pd
import awkward as ak
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from typing import List, Union
from coffea.nanoevents.methods import candidate, vector


def normalize(var: ak.Array, cut: ak.Array = None) -> ak.Array:
    """
    normalize arrays after a cut or selection

    params:
    -------
    var:
        variable array
    cut:
        mask array to filter variable array
    """
    if var.ndim == 2:
        var = ak.firsts(var)
    if cut is None:
        ar = ak.to_numpy(ak.fill_none(var, np.nan))
        return ar
    else:
        ar = ak.to_numpy(ak.fill_none(var[cut], np.nan))
        return ar


def pad_val(
    arr: ak.Array,
    value: float,
    target: int = None,
    axis: int = 0,
    to_numpy: bool = False,
    clip: bool = True,
) -> Union[ak.Array, np.ndarray]:
    """
    pads awkward array up to ``target`` index along axis ``axis`` with value ``value``,
    optionally converts to numpy array
    """
    if target:
        ret = ak.fill_none(
            ak.pad_none(arr, target, axis=axis, clip=clip), value, axis=None
        )
    else:
        ret = ak.fill_none(arr, value, axis=None)
    return ret.to_numpy() if to_numpy else ret


def build_p4(cand: ak.Array) -> ak.Array:
    """
    builds a 4-vector

    params:
    -------
    cand:
        candidate array
    """
    return ak.zip(
        {
            "pt": cand.pt,
            "eta": cand.eta,
            "phi": cand.phi,
            "mass": cand.mass,
            "charge": cand.charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )


def save_dfs_parquet(fname: str, dfs_dict: dict) -> None:
    """
    save dataframes as parquet files
    """
    table = pa.Table.from_pandas(dfs_dict)
    if len(table) != 0:  # skip dataframes with empty entries
        pq.write_table(table, fname + ".parquet")


def ak_to_pandas(output_collection: dict) -> pd.DataFrame:
    """
    cast awkward array into a pandas dataframe
    """
    output = pd.DataFrame()
    for field in ak.fields(output_collection):
        output[field] = ak.to_numpy(output_collection[field])
    return output


def save_output(
    events: ak.Array,
    dataset: str,
    output: pd.DataFrame,
    year: str,
    channel: str,
    output_location: str,
    dir_name: str,
) -> None:
    """
    creates output folders and save dfs to parquet files
    """
    with open("wprime_plus_b/data/simplified_samples.json", "r") as f:
        simplified_samples = json.load(f)
    sample = simplified_samples[year][dataset]
    partition_key = events.behavior["__events_factory__"]._partition_key.replace(
        "/", "_"
    )
    date = datetime.today().strftime("%Y-%m-%d")

    # creating directories for each channel and sample
    if not os.path.exists(
        output_location + date + "/" + dir_name + "/" + year + "/" + channel
    ):
        os.makedirs(
            output_location + date + "/" + dir_name + "/" + year + "/" + channel
        )
    if not os.path.exists(
        output_location
        + date
        + "/"
        + dir_name
        + "/"
        + year
        + "/"
        + channel
        + "/"
        + sample
    ):
        os.makedirs(
            output_location
            + date
            + "/"
            + dir_name
            + "/"
            + year
            + "/"
            + channel
            + "/"
            + sample
        )
    fname = (
        output_location
        + date
        + "/"
        + dir_name
        + "/"
        + year
        + "/"
        + channel
        + "/"
        + sample
        + "/"
        + partition_key
    )
    save_dfs_parquet(fname, output)


def prod_unflatten(array: ak.Array, n: ak.Array):
    """
    Unflattens an array and takes the product through the axis 1

    Parameters:
    -----------
        array: array to unflat
        n: array with the number of objects per event. Used to perform the unflatten operation
    """
    return ak.prod(ak.unflatten(array, n), axis=1)


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


def trigger_match(leptons: ak.Array, trigobjs: ak.Array, trigger_path: str):
    """
    Returns DeltaR matched trigger objects 
    
    leptons:
        electrons or muons arrays
    trigobjs:
        trigger objects array
    trigger_path:
        trigger to match {IsoMu27, Ele35_WPTight_Gsf, Mu50, Mu100}
        
    https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgammaNanoAOD#Trigger_bits_how_to
    """
    match_configs = {
        "IsoMu24": {
            "pt": trigobjs.pt > 22,
            "filterbit": (trigobjs.filterBits & 8) > 0,
            "id": abs(trigobjs.id) == 13
        },
        "IsoMu27": {
            "pt": trigobjs.pt > 25,
            "filterbit": (trigobjs.filterBits & 8) > 0,
            "id": abs(trigobjs.id) == 13
        },
        "Mu50": {
            "pt": trigobjs.pt > 45,
            "filterbit": (trigobjs.filterBits & 1024) > 0,
            "id": abs(trigobjs.id) == 13
        },
        "OldMu100": {
            "pt": trigobjs.pt > 95,
            "filterbit": (trigobjs.filterBits & 2048) > 0,
            "id": abs(trigobjs.id) == 13
        },
        # same as OldMu100?
        # https://github.com/cms-sw/cmssw/blob/CMSSW_10_6_X/PhysicsTools/NanoAOD/python/triggerObjects_cff.py#L79
        "TkMu100": {
            "pt": trigobjs.pt > 95,
            "filterbit": (trigobjs.filterBits & 2048) > 0,
            "id": abs(trigobjs.id) == 13
        },
        "Ele35_WPTight_Gsf": {
            "pt": trigobjs.pt > 33,
            "filterbit": (trigobjs.filterBits & 2) > 0,
            "id": abs(trigobjs.id) == 11
        },
        "Ele32_WPTight_Gsf": {
            "pt": trigobjs.pt > 30,
            "filterbit": (trigobjs.filterBits & 2) > 0,
            "id": abs(trigobjs.id) == 11
        },
        "Ele27_WPTight_Gsf": {
            "pt": trigobjs.pt > 25,
            "filterbit": (trigobjs.filterBits & 2) > 0,
            "id": abs(trigobjs.id) == 11
        },
        "Photon175": {
            "pt": trigobjs.pt > 25,
            "filterbit": (trigobjs.filterBits & 8192) > 0,
            "id": abs(trigobjs.id) == 11
        },
        "Photon200": {
            "pt": trigobjs.pt > 25,
            "filterbit": (trigobjs.filterBits & 8192) > 0,
            "id": abs(trigobjs.id) == 11
        },
        "IsoTkMu24": {
            "pt": trigobjs.pt > 22,
            "filterbit": (trigobjs.filterBits & 8) > 0,
            "id": abs(trigobjs.id) == 13
        },
    }
    pass_pt = match_configs[trigger_path]["pt"]
    pass_id = match_configs[trigger_path]["id"]
    pass_filterbit = match_configs[trigger_path]["filterbit"]
    trigger_cands = trigobjs[pass_pt & pass_id & pass_filterbit]
    delta_r = leptons.metric_table(trigger_cands)
    pass_delta_r = delta_r < 0.1
    n_of_trigger_matches = ak.sum(pass_delta_r, axis=2)
    trig_matched_locs = n_of_trigger_matches >= 1
    return trig_matched_locs


def Z_Vector(muons: ak.Array):

    # num_mask = (ak.num(muons) == 2)
    # muons = muons.mask[num_mask]

    leadings_muons =  ak.pad_none(muons, 2)[:, 0]
    subleading_muons =  ak.pad_none(muons, 2)[:, 1]

    return leadings_muons + subleading_muons


def Boost_ISR_weights(met: ak.Array, Z: ak.Array):
    # source: https://repositorio.uniandes.edu.co/entities/publication/cf6ed855-8ec3-4097-8f0c-9eae967271e8
    # See page 99, eq 7.6
    r_t = -(met.pt*np.cos(Z.phi - met.phi) + Z.pt) 

    return r_t


def pdg_masses():
    
    with open("wprime_plus_b/jsons/wAndtop_masses.json", "r") as f:
        pdg = json.load(f)
                    
    top_mass_pdg = pdg['pdg']['top_mass']           
    w_mass_pdg = pdg['pdg']['w_mass'] 
    
    return top_mass_pdg, w_mass_pdg


def tagger_constants(case: str = "hadronic"):
    # W, top and chi2
    with open("wprime_plus_b/jsons/wAndtop_masses.json", "r") as f:
        masses = json.load(f)


    top_sigma = masses[case]['top_sigma']
    top_low_mass = masses[case]['top_low_mass']
    top_up_mass = masses[case]['top_up_mass']


    w_sigma = masses[case]['w_sigma']
    w_low_mass = masses[case]['w_low_mass']
    w_up_mass = masses[case]['w_up_mass']  

    chi2 = masses[case]['chi2']

    return top_sigma, top_low_mass, top_up_mass, w_sigma, w_low_mass, w_up_mass, chi2


def combine_top_mass_cases(*arrays):
    
    if not arrays:
        raise ValueError("At least one array must be provided")
    
    # Verificar que todos los arrays tengan la misma longitud
    array_lengths = [len(array) for array in arrays]
    if len(set(array_lengths)) != 1:
        raise ValueError("All arrays must have the same length")
    
    # Sumar elemento por elemento
    combined_tops = ak.Array([sum(elements) for elements in zip(*arrays)])
    
    return combined_tops


def combine_top_masks_cases(*masks):
    if not masks:
        raise ValueError("At least one mask must be provided")
    
    #masks = [np.asarray(mask, dtype=bool) for mask in masks]
    
    combined_mask = masks[0]
    for mask in masks[1:]:
        combined_mask = np.logical_or(combined_mask, mask)
    
    return combined_mask

def chi2_test(topJet, wJet, top_sigma, w_sigma, top_mass_pdg,  w_mass_pdg):
        
    t = (topJet.mass - top_mass_pdg) / top_sigma
    w = (wJet.mass - w_mass_pdg) / w_sigma
    

    chi2 = t**2 + w**2
    

    return chi2

def top_tagger(topX, top_tagger_cases=None):
    # Definir todos los escenarios posibles
    escenarios = {
        "case_1": topX.Scenario_1jet_unresolve,
        "case_2": topX.Scenario_2jets_unresolve,
        "case_3": topX.Scenario_2jets_partiallyresolve,
        "case_4": topX.Scenario_3jets_partiallyresolve,
        "case_5": topX.Scenario_3jets_resolve,
        "case_6": topX.Scenario_4jets_resolve,
        "case_7": topX.Scenario_Njets_resolve,
        "case_8": topX.Scenario_Nbjets_resolve,
        "case_9": topX.Scenario_1jet_unresolve_general,
        "case_10": topX.Scenario_2jets_unresolve_general,
        "case_11": topX.Scenario_2jets_partiallyresolve_general,
        "case_12": topX.Scenario_3jets_partiallyresolve_general,
        "case_13": topX.Scenario_3jets_resolve_general,
    }
    
    # Si no se proporcionan casos_a_incluir, se incluyen todos los escenarios
    if top_tagger_cases is None:
        top_tagger_cases = escenarios.keys()
    
    # Aplicar solo los escenarios especificados en casos_a_incluir y recolectar resultados
    tops = {}
    masks = {}
    for key, scenario_func in escenarios.items():
        if key in top_tagger_cases:
            top_result, mask_result = scenario_func()
            tops[key] = top_result
            masks[key] = mask_result
    
    # Combinar máscaras solo de los escenarios incluidos
    mask_top = combine_top_masks_cases(*[masks[key] for key in top_tagger_cases if key in masks])
    
    # Combinar tops solo de los escenarios incluidos
    combined_tops = combine_top_mass_cases(*[tops[key] for key in top_tagger_cases if key in tops])
    
    # Seleccionar tops según la máscara combinada
    selected_tops = combined_tops.mask[mask_top]
    

    return selected_tops, mask_top, masks



def output_metadata(output, weights=None, masks=None, mask_top=None):
    keys = [
        "one_jet_unresolve",                    # Case 1
        "two_jets_unresolve",                   # Case 2
        "two_jets_partially_resolve",           # Case 3
        "three_jets_partially_resolve",         # Case 4
        "three_jets_resolve",                   # Case 5
        "four_jets_resolve",                    # Case 6
        "N_jets_resolve",                       # Case 7
        "N_bjets_resolve",                      # Case 8
        "one_jet_unresolve_gen",                # Case 9
        "two_jets_unresolve_gen",               # Case 10
        "two_jets_partially_resolve_gen",       # Case 11
        "three_jets_partially_resolve_gen",     # Case 12
        "three_jets_resolve_gen"                # Case 13
    ]

    if masks is None:
        output.update({"Total_triggered_nevents":  0.0})
        output.update({"Total_triggered_raw": 0.0})
        output.update({"Total_raw": 0.0})
        output.update({"Total_nevents": 0.0})

        for key in keys:
            output.update({f"{key}_triggered_nevents": 0.0})
            output.update({f"{key}_triggered_raw": 0.0})
            output.update({f"{key}_raw": 0.0})
            output.update({f"{key}_nevents": 0.0})

    else:


        output.update({"Total_triggered_nevents":  ak.sum(weights[mask_top])})
        output.update({"Total_triggered_raw": ak.sum(mask_top)})
        output.update({"Total_raw": ak.sum(mask_top)})
        output.update({"Total_nevents": ak.sum(weights[mask_top])})

        for key in keys:
            case_key = f"case_{keys.index(key) + 1}"
            if case_key in masks:
                case_mask = masks[case_key]

                output.update({f"{key}_triggered_nevents": ak.sum(weights[case_mask])})
                output.update({f"{key}_nevents": ak.sum(weights[case_mask])})
                output.update({f"{key}_triggered_raw": ak.sum(case_mask)})
                output.update({f"{key}_raw": ak.sum(case_mask)})

def histograms_output(
    self,
    bjets, jets, 
    electrons, muons, taus, 
    met, 
    tops,
    mask, 
    lepton_flavor, 
    is_mc,
    events
):
    # Select region objects
    region_bjets = bjets[mask]
    region_jets = jets[mask]
    region_electrons = electrons[mask]
    region_muons = muons[mask]
    region_taus = taus[mask]
    region_met = met[mask]
    region_tops = tops[mask]

    # Define region leptons
    lepton_region_map = {
        "ele": region_electrons,
        "mu": region_muons,
        "tau": region_taus
    }
    region_leptons = lepton_region_map[lepton_flavor]

    # Leading bjets
    leading_bjets = ak.firsts(region_bjets)
    # Lepton-bjet deltaR and invariant mass
    lepton_bjet_dr = leading_bjets.delta_r(region_leptons)
    lepton_bjet_mass = (region_leptons + leading_bjets).mass
    # Lepton-MET transverse mass and deltaPhi
    lepton_met_mass = np.sqrt(
        2.0
        * region_leptons.pt
        * region_met.pt
        * (
            ak.ones_like(region_met.pt)
            - np.cos(region_leptons.delta_phi(region_met))
        )
    )
    lepton_met_delta_phi = np.abs(region_leptons.delta_phi(region_met))
    # Lepton-bJet-MET total transverse mass
    lepton_met_bjet_mass = np.sqrt(
        (region_leptons.pt + leading_bjets.pt + region_met.pt) ** 2
        - (region_leptons + leading_bjets + region_met).pt ** 2
    )


    # HT and  ST variables
    jet_pt_addition = ak.sum(region_jets.pt, axis=1)
    bjet_pt_addition = ak.sum(region_bjets.pt, axis=1)
    tau_pt_addition = ak.sum(region_taus.pt, axis=1)
    lepton_pt_addition = ak.sum(region_leptons.pt, axis = 1)  # Depending of the channel, we will have muons, taus, or electrons.
    muon_pt_addition = ak.sum(region_muons.pt, axis=1)
    electron_pt_addition = ak.sum(region_electrons.pt, axis=1)

    region_HT = jet_pt_addition + bjet_pt_addition 
    region_ST = lepton_pt_addition+ region_HT
    region_ST_met = lepton_pt_addition + region_HT + region_met.pt
    region_ST_full = region_ST + muon_pt_addition + electron_pt_addition + tau_pt_addition 
    
    # Add features to the object (assumed to have a method `add_feature`)
    self.add_feature("lepton_pt", region_leptons.pt)
    self.add_feature("lepton_eta", region_leptons.eta)
    self.add_feature("lepton_phi", region_leptons.phi)

    if is_mc:
        # genPartFlav is only defined in MC samples
        self.add_feature("genPartFlav", region_taus.genPartFlav)
    self.add_feature("decayMode", region_taus.decayMode)
    self.add_feature("isolation_electrons", region_taus.idDeepTau2017v2p1VSe)
    self.add_feature("isolation_jets", region_taus.idDeepTau2017v2p1VSjet)
    self.add_feature("isolation_muons", region_taus.idDeepTau2017v2p1VSmu)

    self.add_feature("bjet_pt", region_bjets.pt)
    self.add_feature("bjet_eta", region_bjets.eta)
    self.add_feature("bjet_phi", region_bjets.phi)

    self.add_feature("jet_pt", region_jets.pt)
    self.add_feature("jet_eta", region_jets.eta)
    self.add_feature("jet_phi", region_jets.phi)
    
    
    self.add_feature("met", region_met.pt)
    self.add_feature("met_phi", region_met.phi)
    
    self.add_feature("lepton_bjet_dr", lepton_bjet_dr)
    self.add_feature("lepton_bjet_mass", lepton_bjet_mass)
    
    self.add_feature("lepton_met_mass", lepton_met_mass)
    self.add_feature("lepton_met_delta_phi", lepton_met_delta_phi)
    self.add_feature("lepton_met_bjet_mass", lepton_met_bjet_mass)
    
    self.add_feature("njets", ak.num(region_jets))
    self.add_feature("nbjets", ak.num(region_bjets))
    self.add_feature("npvs", events.PV.npvsGood[mask])
    self.add_feature("nmuons", ak.num(region_muons))
    self.add_feature("nelectrons", ak.num(region_electrons))
    self.add_feature("ntaus", ak.num(region_taus))

    self.add_feature("HT", region_HT)    
    self.add_feature("ST", region_ST)  
    self.add_feature("ST_met", region_ST_met)  
    self.add_feature("ST_full", region_ST_full)

    
    self.add_feature("top_mrec", region_tops)



def histograms_output_v2(
    self,
    bjets, jets, 
    electrons, muons, taus, 
    met, 
    tops,
    mask, 
    lepton_flavor, 
    events
):
    # Select region objects
    region_bjets = bjets[mask]
    region_jets = jets[mask]
    region_electrons = electrons[mask]
    region_muons = muons[mask]
    region_taus = taus[mask]
    region_met = met[mask]
    region_tops = tops[mask]

    # Define region leptons
    lepton_region_map = {
        "ele": region_electrons,
        "mu": region_muons,
        "tau": region_taus
    }
    region_leptons = lepton_region_map[lepton_flavor]

    # Leading bjets
    leading_bjets = ak.firsts(region_bjets)
    # Lepton-bjet deltaR and invariant mass
    lepton_bjet_dr = leading_bjets.delta_r(region_leptons)
    lepton_bjet_mass = (region_leptons + leading_bjets).mass
    # Lepton-MET transverse mass and deltaPhi
    lepton_met_mass = np.sqrt(
        2.0
        * region_leptons.pt
        * region_met.pt
        * (
            ak.ones_like(region_met.pt)
            - np.cos(region_leptons.delta_phi(region_met))
        )
    )
    lepton_met_delta_phi = np.abs(region_leptons.delta_phi(region_met))
    # Lepton-bJet-MET total transverse mass
    lepton_met_bjet_mass = np.sqrt(
        (region_leptons.pt + leading_bjets.pt + region_met.pt) ** 2
        - (region_leptons + leading_bjets + region_met).pt ** 2
    )

    # Add features to the object (assumed to have a method `add_feature`)
    self.add_feature("lepton_pt", region_leptons.pt)
    self.add_feature("lepton_eta", region_leptons.eta)
    self.add_feature("lepton_phi", region_leptons.phi)
    self.add_feature("jet_pt", leading_bjets.pt)
    self.add_feature("jet_eta", leading_bjets.eta)
    self.add_feature("jet_phi", leading_bjets.phi)
    self.add_feature("met", region_met.pt)
    self.add_feature("met_phi", region_met.phi)
    self.add_feature("lepton_bjet_dr", lepton_bjet_dr)
    self.add_feature("lepton_bjet_mass", lepton_bjet_mass)
    self.add_feature("lepton_met_mass", lepton_met_mass)
    self.add_feature("lepton_met_delta_phi", lepton_met_delta_phi)
    self.add_feature("lepton_met_bjet_mass", lepton_met_bjet_mass)
    self.add_feature("njets", ak.num(region_jets))
    self.add_feature("npvs", events.PV.npvsGood[mask])
    self.add_feature("top_mrec", region_tops)
#    self.add_feature("HT", events.LHE.HT[mask])    
