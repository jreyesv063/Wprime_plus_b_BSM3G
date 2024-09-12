import json
import numpy as np
import awkward as ak
import importlib.resources
from pathlib import Path
from coffea.nanoevents.methods.base import NanoEventsArray


def select_good_ditaus(
    taus: ak.Array,
    passing_first_tau: str = "Tight",
    passing_second_tau: str = "Loose",
    failing_second_tau: str = "Tight",
) -> ak.highlevel.Array:
    """
    Selects tau pairs based on WP criteria and charge.

    Parameters:
    taus -- Array of taus.
    charge_selection -- Charge selection ("LS" or "OS").
    passing_first_tau -- WP that the first tau should pass.
    passing_second_tau -- WP that the second tau should pass.
    failing_second_tau -- WP that the second tau should fail


    Input taus passed loose wp, Loose is 8; Tight is 32, so a tau pass Loose, pass Tigh given that we use as criteria TauvsJet > wp

    Tau 1: Pass Tight.
    Tau 2: Pass Loose and Fail Tight

    
    Q(tau1)Q(tau2) > 0 or OS

    Returns:
    A boolean array indicating whether each tau pair meets the criteria.

    """

    with importlib.resources.open_text("wprime_plus_b.data", "tau_wps.json") as file:
        taus_wps = json.load(file)


    if failing_second_tau == "None":
        first_lepton_wp_pass = taus_wps["DeepTau2017"]["deep_tau_jet"][passing_first_tau]
        second_lepton_wp_pass = taus_wps["DeepTau2017"]["deep_tau_jet"][passing_second_tau]
    else:
        first_lepton_wp_pass = taus_wps["DeepTau2017"]["deep_tau_jet"][passing_first_tau]
        second_lepton_wp_pass = taus_wps["DeepTau2017"]["deep_tau_jet"][passing_second_tau]
        second_lepton_wp_fail = taus_wps["DeepTau2017"]["deep_tau_jet"][failing_second_tau]


    # Ensure there are at least two taus
    #if ak.num(taus) < 2:
    #    return ak.zeros_like(taus, dtype=bool)
    
    # Select two taus
    leading_tau = ak.pad_none(taus, 2)[:, 0]
    subleading_tau = ak.pad_none(taus, 2)[:, 1]
    

    if failing_second_tau == "None":
         good_ditau = (
            (leading_tau.idDeepTau2017v2p1VSjet > first_lepton_wp_pass)
            & (subleading_tau.idDeepTau2017v2p1VSjet > second_lepton_wp_pass)
         )

    else:
    # Apply WP criteria for all possible cases
        good_ditau = (

            # Case 1: Leading tau passes Tight and subleading tau passes Loose but fails Tight
            (
                (leading_tau.idDeepTau2017v2p1VSjet > first_lepton_wp_pass)
                & (subleading_tau.idDeepTau2017v2p1VSjet > second_lepton_wp_pass)
                & (subleading_tau.idDeepTau2017v2p1VSjet < second_lepton_wp_fail)
            )
            
            ^  # XOR operation: output True when either of the operands is True and the other one is False, but False when both operands are True or False (the same).
            
            # Case 2: Subleading tau passes Tight and leading tau passes Loose but fails Tight
            (
                (subleading_tau.idDeepTau2017v2p1VSjet > first_lepton_wp_pass)
                & (leading_tau.idDeepTau2017v2p1VSjet > second_lepton_wp_pass)
                & (leading_tau.idDeepTau2017v2p1VSjet < second_lepton_wp_fail)
            )
        )

    return good_ditau 
