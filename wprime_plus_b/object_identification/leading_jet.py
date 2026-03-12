import json
import pickle
import numpy as np
import awkward as ak

def get_leading_jet_mask(
    events: ak.Array, 
    jets: ak.Array, 
    leading_pt_threshold: float = 0.2, 
    syst_var: bool = True
):

    leading_jet = ak.firsts(jets)

    if (
        syst_var
        and hasattr(events, "genWeight")
        and hasattr(events.Jet, "JES_pt_up")
        and hasattr(events.Jet, "JES_pt_down")
        and hasattr(events.Jet, "JER_pt_up")
        and hasattr(events.Jet, "JER_pt_down")
    ):  
        # JES
        leading_jet_JES_up_mask = (
            (leading_jet.JES_pt_up >= leading_pt_threshold)
        )

        leading_jet_JES_down_mask = (
            (leading_jet.JES_pt_down >= leading_pt_threshold)
        )

        # JER
        leading_jet_JER_up_mask = (
            (leading_jet.JER_pt_up >= leading_pt_threshold)
        )

        leading_jet_JER_down_mask = (
            (leading_jet.JER_pt_down >= leading_pt_threshold)
        )


        return {
            "nominal": (leading_jet.pt > leading_pt_threshold),
            "JES": {
                "up":  leading_jet_JES_up_mask, "down":  leading_jet_JES_down_mask
            },
            "JER": {
                "up":  leading_jet_JER_up_mask, "down":  leading_jet_JER_down_mask
            }
        }


    else:   
        return {
            "nominal": (leading_jet.pt > leading_pt_threshold)
        }

