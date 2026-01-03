import json
import numpy as np
import awkward as ak
from typing import Dict
import importlib.resources


def select_good_jets(
    events: ak.Array,
    jets: ak.Array,    
    year: str = "2017",
    jet_pt_threshold: int = 20,
    jet_eta_threshold: float = 2.4,
    jet_id_wp: str = "TightLepVeto",
    jet_pileup_id: str = "Tight",
    leading_jet_pt: float = 100.0,
    is_mc: bool = True,
) -> Dict[str, ak.Array]:
    """
    Selects and filters 'good' b-jets from a collection of jets based on specified criteria

    Parameters:
    -----------
    events:
        A collection of events represented using the NanoEventsArray class.

    year: {'2016', '2017', '2018'}
        Year for which the data is being analyzed. Default is '2017'.

    btag_working_point: {'L', 'M', 'T'}
        Working point for b-tagging. Default is 'M'.

    jet_id: https://twiki.cern.ch/twiki/bin/view/CMS/JetID#Run_II
        Jet ID flags {1, 2, 3, 6, 7}
        For 2016 samples:
            1 means: pass loose ID, fail tight, fail tightLepVeto
            3 means: pass loose and tight ID, fail tightLepVeto
            7 means: pass loose, tight, tightLepVeto ID.
        For 2017 and 2018 samples:
            2 means: pass tight ID, fail tightLepVeto
            6 means: pass tight and tightLepVeto ID.

    jet_pileup_id: https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJetID
        Pileup ID flags for pre-UL trainings {0, 4, 6, 7}. Should be applied only to AK4 CHS jets with pT < 50 GeV
        0 means 000: fail all PU ID;
        4 means 100: pass loose ID, fail medium, fail tight;
        6 means 110: pass loose and medium ID, fail tight;
        7 means 111: pass loose, medium, tight ID.

    Returns:
    --------
        An Awkward Array mask containing the selected "good" b-jets that satisfy the specified criteria.
    """

    # --------------------------------------------------
    # Working points and flags
    # --------------------------------------------------

    puid_wps = {"Fail": 0, "Loose": 4, "Medium": 6, "Tight": 7}

    if year in ["2016APV","2016", "2017", "2018"]:
        # --------------------------------------------------------
        # Run 2: jetId flags are stored correctly in NanoAOD
        # --------------------------------------------------------        
        jet_id_flags = {
            "2016APV": {"Loose": 1, "Tight": 3, "TightLepVeto": 6},
            "2016": {"Loose": 1, "Tight": 3, "TightLepVeto": 6},
            "2017": {"Tight": 2, "TightLepVeto": 6},
            "2018": {"Tight": 2, "TightLepVeto": 6},
        }


        jet_id = jet_id_flags[year][jet_id_wp]

    # --------------------------------------------------
    # Load b-tag thresholds
    # --------------------------------------------------
    with open("wprime_plus_b/json_files/btagWPs.json", "r") as f:
        btag_thresholds = json.load(f)["deepJet"][year]
        btag_fail = (
            btag_thresholds[btag_working_point_fail]
            if btag_working_point_fail
            else None
        )

    # --------------------------------------------------
    # Helper to build masks
    # --------------------------------------------------

    def build_masks(shift):
        # Low pT jets (PU ID applied)
        low_pt_mask = (
            (shift.pt > jet_pt_threshold)
            & (shift.pt < 50)
            & (np.abs(shift.eta) < jet_eta_threshold)
            & (shift.jetId  == jet_id)
            & (shift.puId  == puid_wps[jet_pileup_id])
        )

        # High pT jets (no PU ID)
        high_pt_mask = (
            (shift.pt >= 50)
            & (np.abs(shift.eta) < jet_eta_threshold)
            & (shift.jetId  == jet_id)
        )


        return low_pt_mask | high_pt_mask


    # --------------------------------------------------
    # Main logic
    # --------------------------------------------------
    good_jet_masks = {}

    if is_mc:
        jet_shifts = {
            "nominal": jets,
            "JES_up": jets.JES_jes.up,
            "JES_down": jets.JES_jes.down,
            "JER_up": jets.JER.up,
            "JER_down": jets.JER.down,
        }

        for name, shift in jet_shifts.items():
            good_jet_masks[name] = build_masks(shift)

    else:
        good_jet_masks["nominal"] = build_masks(jets)

    return good_jet_masks