import json
import numpy as np
import awkward as ak

from wprime_plus_b.object_identification.utils import delta_r


def select_good_Z(
    objects: list,
    lepton_flav: str,
    mass_min: float,
    mass_max: float,
    charge_selection: str,
    cross_cleaning: float,
    syst_var: bool
):

    # ============================================================
    # Leading and subleading leptons
    # ============================================================   
    lepton_map = {
        "ele": "electrons",
        "mu": "muons",
        "tau": "taus"
    }
    leptons = objects[lepton_map[lepton_flav]]

    leading_lepton = ak.pad_none(leptons, 2)[:, 0]
    subleading_lepton = ak.pad_none(leptons, 2)[:, 1]


    # ============================================================
    # Store Z boson variables
    # ============================================================ 
    # Z boson variables
    Z_mass = (leading_lepton + subleading_lepton).mass
    Z_charge = (leading_lepton.charge + subleading_lepton.charge)
    Z_pt = (leading_lepton + subleading_lepton).pt

    objects["events"] = ak.with_field(objects["events"], leading_lepton.pt, "leading_pt")
    objects["events"] = ak.with_field(objects["events"], subleading_lepton.pt, "subleading_pt")
    
    objects["events"] = ak.with_field(objects["events"], Z_mass, "Z_mass")
    objects["events"] = ak.with_field(objects["events"], Z_charge, "Z_charge")
    objects["events"] = ak.with_field(objects["events"], Z_pt, "Z_pt")


    # ============================================================
    # Charge
    # ============================================================    
    charge = {
        "OS": leading_lepton.charge * subleading_lepton.charge < 0,
        "LS": leading_lepton.charge * subleading_lepton.charge > 0
    }

    Ql1_Ql2_mask = charge[charge_selection]
    
    # ============================================================
    # Mass
    # ============================================================  
    # Mass: Invariant mass https://coffea-hep.readthedocs.io/en/v0.7.23/api/coffea.nanoevents.methods.vector.LorentzVector.html#coffea.nanoevents.methods.vector.LorentzVector.mass
    mass_mask = (
        ((leading_lepton + subleading_lepton).mass >=  mass_min)
        & ((leading_lepton + subleading_lepton).mass <=  mass_max)
    )
    
    # Delta R selection
    cross_cleaning = delta_r(leading_lepton, subleading_lepton, threshold = cross_cleaning)

    # ============================================================
    # Final electron selection mask
    # ============================================================   
    Z_mask_ref =  Ql1_Ql2_mask & cross_cleaning
    Z_mask = Z_mask_ref & mass_mask

    
    # ============================================================
    # Systematic variations: 
    # ============================================================     
    if syst_var and hasattr(objects["events"], "genWeight"):
        if hasattr(leptons, "pt_up") and hasattr(leptons, "pt_down"):
            # Up: ak.with_field changes the pt field to pt_up, allowing you to use (leading + subleading).mass with the up variation.
            leading_lepton_up = ak.pad_none(ak.with_field(leptons, leptons.pt_up, "pt"), 2)[:,0]
            subleading_lepton_up = ak.pad_none(ak.with_field(leptons, leptons.pt_up, "pt"), 2)[:,1]
            
            mass_mask_up = (
                ((leading_lepton_up + subleading_lepton_up).mass >=  mass_min)
                & ((leading_lepton_up + subleading_lepton_up).mass <=  mass_max)
            )
        
            # Down: ak.with_field changes the pt field to pt_up, allowing you to use (leading + subleading).mass with the up variation.
            leading_lepton_down = ak.pad_none(ak.with_field(leptons, leptons.pt_down, "pt"), 2)[:,0]
            subleading_lepton_down = ak.pad_none(ak.with_field(leptons, leptons.pt_down, "pt"), 2)[:,1]        
    
            mass_mask_down = (
                ((leading_lepton_down + subleading_lepton_down).mass >=  mass_min)
                & ((leading_lepton_down + subleading_lepton_down).mass <=  mass_max)
            )

            Z_masks = {
                "nominal": ak.fill_none(Z_mask, False),
                "up": ak.fill_none(Z_mask_ref & mass_mask_up, False),
                "down": ak.fill_none(Z_mask_ref & mass_mask_down, False)
            }

    else:

        Z_masks = {"nominal": ak.fill_none(Z_mask, False)}

    return Z_masks, objects
        
