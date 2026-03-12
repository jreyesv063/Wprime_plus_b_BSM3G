import json
import numpy as np
import correctionlib
import awkward as ak
from typing import Type
from coffea.analysis_tools import Weights
from wprime_plus_b.corrections.utils import get_pog_json

from wprime_plus_b.processors.utils.analysis_utils import delta_r_mask


def jetvetomaps_mask(events: ak.Array, year: str):
    """
    source: https://cms-jerc.web.cern.ch/JVM/
    
    These are the jet veto maps showing regions with an excess of jets (hot zones) and lack of jets
    (cold zones). 
    
    - Hot zones: Towers with sufficient data (too much energy).
    - Cold zones:  Towers with low data (energy loss).
    
    The use of phi and eta allows the areas to be located.
    
    Strategy (Data and MC):

        1) Load the correction.
        2) Loop ver all AK4 in the event.
        3) Determine phi and eta for each jet.
        4) Obtain the value using the correction:
            - 0 -> tower is allowed.
            - 1 -> tower ius vetoed.
        5) Important: If any selected jet falls in a vetoed tower: reject the event.


        Run 2/3 recommendations: https://cms-jerc.web.cern.ch/Recommendations/#2025_2

        - Run 2:
            * jet pT > 15 GeV
            * (jet charged EM fraction + jet neutral EM fraction) < 0.9
            * jets that don’t overlap with PF muon (dR < 0.2)
            * tight jet ID
            * PU jet ID for CHS jets with pT < 50 GeV
            
            

        - Run 3:
           * jet pT > 15 GeV
           * (jet charged EM fraction + jet neutral EM fraction) < 0.9 
           * tightLepVeto 
           * jet ID (in v15 NanoAOD, the Jet_jetId branch is not available. To apply jet ID selections manually, please refer to the JetID page).
           
    """

    # =====================================================================
    #  Read json file: correction
    # =====================================================================
    with open("wprime_plus_b/corrections/correction_names/JME.json", "r") as f:
        case = json.load(f)

    correction_name = case['jet_veto_maps'][year]

    with open("wprime_plus_b/json_files/jet.json", "r") as f:
        jet_info = json.load(f)

    # ====================================================================
    # Conditions for applying the correction: See Run 2/3 recommendations
    # =====================================================================    
 
    nj = ak.num(events.Jet)
    jets  = events.Jet
    muons = events.Muon

    
    corr_mask = (
        (jets.pt > 15.0)
        & ((jets.chEmEF + jets.neEmEF) < 0.9)
    )

    if year in ["2016APV", "2016", "2017", "2018"]:
        corr_mask = (
            corr_mask 
            & (~delta_r_mask(jets, muons, 0.2))
            & (jets.jetId >= jet_info["jetid"][year]["Tight"])
        )

        corr_mask = ak.where(
            (jets.pt < 50),
            (corr_mask & (jets.puId == jet_info["pujetid"][year]["Tight"])),
            (corr_mask)
        )
        

    elif year in ["2022_pre", "2022_post", "2023_pre", "2023_post", "2024"]:
        corr_mask = corr_mask & (jets.jetId == jet_info["pujetid"][year]["TightLepVeto"])

    # =====================================================================
    #  Jet candidates
    # =====================================================================     
    in_jet_mask = (
        corr_mask
        & (np.abs(jets.eta) < 5.19)
        & (np.abs(jets.phi) < 3.14)
    )

    in_jets = jets.mask[in_jet_mask]

    jets_eta = ak.flatten(ak.fill_none(in_jets.eta, 0.0))
    jets_phi = ak.flatten(ak.fill_none(in_jets.phi, 0.0))

    
    # =====================================================================
    # Correction: Obtain vetomaps
    # =====================================================================
    cset = correctionlib.CorrectionSet.from_file(get_pog_json("jetvetomaps", year))
    
    maps = cset[correction_name].evaluate("jetvetomap", jets_eta, jets_phi)

    # Remove arbitrary values obtained due to None
    vetomaps = ak.where(ak.flatten(in_jet_mask), maps, 0)

    
    # veto == 0 → good region, event pass
    jet_is_good = ak.unflatten( 
        (vetomaps == 0), 
        nj
    )

    # Select valid events
    event_mask = ak.all(jet_is_good, axis=1)

    
    return event_mask