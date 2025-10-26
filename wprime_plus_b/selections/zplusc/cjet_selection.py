import json
import numpy as np
import awkward as ak
import importlib.resources


def select_good_cjets(
    jets,
    year: str = "2017",
    ctag_working_point: str = "M",
    jet_pt_threshold: int = 20,
    jet_eta_threshold: float = 2.4,
    jet_id_wp: str = "tight_tightLepVeto",
    jet_pileup_id: str = "T",
    is_mc: bool = True,
) -> ak.highlevel.Array:
    """
    Selects and filters 'good' c-jets from a collection of jets based on specified criteria

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
        An Awkward Array mask containing the selected "good" c-jets that satisfy the specified criteria.
    """
   
    puid_wps = {
        "fail": 0,
        "L": 4,
        "M": 6,
        "T": 7,
    }
    jet_id_flags = {
        "2016APV": {
            "loose": 1,
            "tight": 3,
            "tightLepVeto": 6,
        },
        "2016": {
            "loose": 1,
            "tight": 3,
            "tightLepVeto": 6,
        },       
        "2017": {
            "tight": 2,
            "tightLepVeto": 6,
        },
        "2018": {
            "tight": 2,
            "tightLepVeto": 6,
        }
    }

    jet_id = jet_id_flags[year][jet_id_wp]


    discriminator = "CvB_cut"
    opposite_discriminator = "CvL_cut"

    # open and load ctagDeepFlavB working point
    with importlib.resources.open_text("wprime_plus_b.data", "ctagWPs.json") as file:
        ctagwps = json.load(file)
        ctag_threshold = (
                        ctagwps["deepJet"][year][discriminator][ctag_working_point], 
                        ctagwps["deepJet"][year][opposite_discriminator][ctag_working_point]
                        )


    good_cjet_masks = {}
    

    if is_mc:
        jet_shift = {
            "JES": {"nominal": jets,
                    # "up":  jets.JES_jes.up,
                    # "down": jets.JES_jes.down
            },
            # "JER": {
            #         "up":  jets.JER.up,
            #         "down": jets.JER.down
            # },  
        }



        for shift_type, variations in jet_shift.items():
            for variation, shift in variations.items():
                # Máscara para jets de bajo pT
                low_pt_jets_mask = (
                    (shift.pt > jet_pt_threshold) & (shift.pt < 50)
                    & (np.abs(shift.eta) < jet_eta_threshold)
                    & (shift.jetId == jet_id)
                    & (shift.puId == puid_wps[jet_pileup_id])
                    & (shift.btagDeepFlavCvB > ctag_threshold[0])
                    & (shift.btagDeepFlavCvL > ctag_threshold[1])
                )

                # Máscara para jets de alto pT
                high_pt_jets_mask = (
                    (shift.pt >= 50)
                    & (np.abs(shift.eta) < 2.4)
                    & (shift.jetId == jet_id)
                    & (shift.btagDeepFlavCvB > ctag_threshold[0])
                    & (shift.btagDeepFlavCvL > ctag_threshold[1])
                )

                # Guardar las máscaras en el diccionario
                if variation == "nominal":
                    good_cjet_masks[variation] = ak.where(
                        (shift.pt > jet_pt_threshold) & (shift.pt < 50),
                        low_pt_jets_mask,
                        high_pt_jets_mask,
                    )
                else:
                    good_cjet_masks[f"{shift_type}_{variation}"] = ak.where(
                        (shift.pt > jet_pt_threshold) & (shift.pt < 50),
                        low_pt_jets_mask,
                        high_pt_jets_mask,
                    )
                
    else:
        low_pt_jets_mask = (
            (jets.pt > jet_pt_threshold)
            & (jets.pt < 50)
            & (np.abs(jets.eta) < jet_eta_threshold)
            & (jets.jetId == jet_id)
            & (jets.puId == puid_wps[jet_pileup_id])
            & (jets.btagDeepFlavCvB > ctag_threshold[0])
            & (jets.btagDeepFlavCvL > ctag_threshold[1])
        )

        high_pt_jets_mask = (
            (jets.pt >= 50)
            & (np.abs(jets.eta) < 2.4)
            & (jets.jetId == jet_id)
            & (jets.btagDeepFlavCvB > ctag_threshold[0])
            & (jets.btagDeepFlavCvL > ctag_threshold[1])
        )

        good_cjet_masks["nominal"] = ak.where(
            (jets.pt > jet_pt_threshold) & (jets.pt < 50),
            low_pt_jets_mask,
            high_pt_jets_mask,
        )


    return good_cjet_masks