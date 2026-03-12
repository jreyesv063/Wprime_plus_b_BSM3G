import json
import copy
import correctionlib
import numpy as np
import awkward as ak
import importlib.resources
from typing import Type
from pathlib import Path
#from .utils import unflat_sf
from coffea.analysis_tools import Weights
from wprime_plus_b.corrections.utils import pog_years, get_pog_json, unflat_sf


# ----------------------------------
# lepton scale factors
# -----------------------------------
#
# Electron
#    - ID: 
#          -  cutbased ID: Loose, Medium, Tight, Veto
#          -  mva: wp80iso, wp80noiso, wp90iso, wp90noiso
#    - Reco: RecoAbove20, RecoBelow20

class ElectronCorrector:
    """
    Electron corrector class

    Parameters:
    -----------
    electrons:
        electron collection
    hlt:
        high level trigger branch
    weights:
        Weights object from coffea.analysis_tools
    year:
        Year of the dataset {'2016', '2017', '2018'}
    variation:
        if 'nominal' (default) add 'nominal', 'up' and 'down'
        variations to weights container. else, add only 'nominal' weights.
    """

    def __init__(
        self,
        electrons: ak.Array,
        weights: Type[Weights],
        year: str = "2017",
        electron_mask = None
    ) -> None:
        
        self.electrons, self.nevents  = electrons, len(electrons)

        # flat electrons array
        self.e, self.n = ak.flatten(electrons), ak.num(electrons)

        # weights container
        self.electron_mask = electron_mask
        self.weights = weights

        # define correction set
        self.cset = correctionlib.CorrectionSet.from_file(get_pog_json(json_name="electron", year=year))
        self.year = year
        self.pog_year = pog_years[year]

        # ===========================================================
        #  Read json file: corrections
        # ============================================================
        # Correction name
        with open("wprime_plus_b/corrections/correction_names/EGM.json", "r") as f:
            self.case = json.load(f)

            
 
    def add_id_weight(self, id_working_point: str) -> None:
        """
        add electron identification scale factors to weights container

        Parameters:
        -----------
            id_working_point:
                Working point {'Loose', 'Medium', 'Tight', 'wp80iso', 'wp80noiso', 'wp90iso', 'wp90noiso'}
        """
        # ===========================================================
        #  Read json file: electron id
        # ============================================================
        # Correction name
        with open("wprime_plus_b/json_files/electron.json", "r") as f:
            e_id_map = json.load(f)['Id']


        case = "MVA" if id_working_point.startswith("wp") else "Cutbased"
        if case == "MVA":            
            electron_id_mask = getattr(self.e, e_id_map[case][id_working_point])
        else :
            electron_id_mask = getattr(self.e, e_id_map[case]["flag"]) == self.e_id[case][id_working_point]
            

        # remove '_UL' from year
        year = self.pog_year.replace("_UL", "")
        correction_name = self.case["CMS_eff_e_id_13TeV"][self.year]
        
        # =============================================================
        #  Electron candidates
        # =============================================================    
        # get 'in-limits' electrons
        electron_pt_mask = ((self.e.pt > 10.0) & (self.e.pt < 499.999))  # potential problems with pt > 500 GeV
        in_electron_mask = electron_pt_mask & electron_id_mask
        in_electrons = self.e.mask[in_electron_mask]
        
        # get electrons transverse momentum and pseudorapidity (replace None values with some 'in-limit' value)
        electron_pt = ak.fill_none(in_electrons.pt, 10.0)
        electron_eta = ak.fill_none(in_electrons.eta, 0.0)
               
        # =============================================================
        # Correction: event-level weight (nominal/up/down)
        # =============================================================        
        # Get nominal, up, and down scale factors
        nominal_sf, up_sf, down_sf = [
            unflat_sf(self.cset[correction_name].evaluate(year, v, id_working_point, electron_eta, electron_pt), in_electron_mask, self.n)
            for v in ("sf", "sfup", "sfdown")
        ]

        nominal_sf, up_sf, down_sf = [
            ak.where(self.electron_mask, sf, 1.0)
            for sf in (nominal_sf, up_sf, down_sf)
        ]
        
        # add scale factors to weights container
        self.weights.add(
            name=f"CMS_eff_e_id_13TeV_{self.year}",
            weight=nominal_sf,
            weightUp=up_sf,
            weightDown=down_sf,
        )
       
    
    
    def add_reco_weight(self, reco_case: str) -> None:
        """add electron reconstruction scale factors to weights container"""

        # Mask, default value
        reco_list = {
            "Above20": ((self.e.pt > 20.0), 21.0),
            "Below20": (((self.e.pt > 10.0) & (self.e.pt < 20.0)), 19.0),
            "20to75":  (((self.e.pt > 20.0) & (self.e.pt < 75.0)), 21.0),
            "Above75": ((self.e.pt > 75.0), 76.0)
        }
        
        # remove '_UL' from year
        year = self.pog_year.replace("_UL", "")
        correction_name = self.case[f"CMS_eff_e_reco_{reco_case}_13TeV"][self.year]

        
        # =============================================================
        #  Electron candidates
        # =============================================================  
        # get 'in-limits' electrons
        electron_pt_mask, pt_fill = reco_list[reco_case][0], reco_list[reco_case][1] 
        in_electron_mask = electron_pt_mask
        in_electrons = self.e.mask[in_electron_mask]
        
        # get electrons transverse momentum and pseudorapidity (replace None values with some 'in-limit' value)
        electron_pt = ak.fill_none(in_electrons.pt, pt_fill)
        electron_eta = ak.fill_none(in_electrons.eta, 0.0)


        # =============================================================
        # Correction: event-level weight (nominal/up/down)
        # =============================================================  
        # Get nominal, up, and down scale factors
        nominal_sf, up_sf, down_sf = [
            unflat_sf(self.cset[correction_name].evaluate(year, v, f"Reco{reco_case}", electron_eta, electron_pt), in_electron_mask, self.n)
            for v in ("sf", "sfup", "sfdown")
        ]        
        

        nominal_sf, up_sf, down_sf = [
            ak.where(self.electron_mask, sf, 1.0)
            for sf in (nominal_sf, up_sf, down_sf)
        ]

        # add scale factors to weights container
        self.weights.add(
            name=f"CMS_eff_e_reco_{reco_case}_13TeV_{self.year}",
            weight=nominal_sf,
            weightUp=up_sf,
            weightDown=down_sf,
        )