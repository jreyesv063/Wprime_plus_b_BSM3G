import json
import copy
import numpy as np
import awkward as ak
import correctionlib
from typing import Type
from pathlib import Path
import importlib.resources
from .utils import unflat_sf
from coffea.analysis_tools import Weights
from wprime_plus_b.corrections.utils import pog_years, get_pog_json

# https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2016
# https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2017
# https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2018

class MuonCorrector:
    """

    Source: https://muon-wiki.docs.cern.ch/guidelines/corrections/?h=l1#l1-trigger-prefiring
    
    Muon corrector class

    Parameters:
    -----------
    muons:
        muons collection
    weights:
        Weights object from coffea.analysis_tools
    year:
        Year of the dataset {'2016', '2017', '2018'}
    variation:
        syst variation
    id_wp:
        ID working point {'loose', 'medium', 'tight'}
    iso_wp:
        Iso working point {'loose', 'medium', 'tight'}
    """

    def __init__(
        self,
        muons: ak.Array,
        weights: Type[Weights],
        year: str = "2017",
        id_wp: str = "Tight",
        iso_wp: str = "Tight",
        variation: str = "nominal",
        pt_range: str = "MediumPt",
        muon_mask = None
    ) -> None:
        self.muons = muons
        self.variation = variation
        self.id_wp = id_wp
        self.iso_wp = iso_wp
        
        # muon array
        self.muons = muons
        
        # flat muon array
        self.m, self.n = ak.flatten(muons), ak.num(muons)
        
        # weights container
        self.muon_mask = muon_mask
        self.weights = weights
        
        # define correction set
        self.cset = correctionlib.CorrectionSet.from_file(
            get_pog_json(json_name="muon", year=year)
        )
        self.year = year
        self.pog_year = pog_years[year]

        # ===========================================================
        #  Read json file: corrections
        # ============================================================
        # Correction name
        with open("wprime_plus_b/corrections/correction_names/MUO.json", "r") as f:
            self.case = json.load(f)[pt_range]

        # ===========================================================
        #  Read json file: muon id, muon reco
        # ============================================================
        # Correction name
        with open("wprime_plus_b/json_files/muon.json", "r") as f:
            self.muon_map = json.load(f)
            

    def add_reco_weight(self):
        """
        add muon RECO scale factors to weights container
        """
        
        correction_name = self.case["CMS_eff_m_reco"][self.year]

        # =============================================================
        #  Muon candidates
        # =============================================================      
        # get muons within SF binning
        muon_pt_mask = self.m.pt >= 40.0
        muon_eta_mask = np.abs(self.m.eta) < 2.4
        in_muon_mask = muon_pt_mask & muon_eta_mask 
        in_muons = self.m.mask[in_muon_mask]

        # get muons pT and abseta (replace None values with some 'in-limit' value)
        muon_pt = ak.fill_none(in_muons.pt, 40.0)
        muon_eta = np.abs(ak.fill_none(in_muons.eta, 0.0))


        # =============================================================
        # Correction: event-level weight (nominal/up/down)
        # =============================================================         
        # Get nominal, up, and down scale factors
        nominal_sf, up_sf, down_sf = [
            unflat_sf(self.cset[correction_name].evaluate(muon_eta, muon_pt, v), in_muon_mask, self.n)
            for v in ("nominal", "systup", "systdown")
        ]


        nominal_sf, up_sf, down_sf = [
            ak.where(self.muon_mask, sf, 1.0)
            for sf in (nominal_sf, up_sf, down_sf)
        ]
        
        # add scale factors to weights container
        self.weights.add(
            name=f"CMS_eff_m_reco_syst_{self.year}",
            weight=nominal_sf,
            weightUp=up_sf,
            weightDown=down_sf,
        )
            

    def add_id_weight(self):
        """
        add muon ID scale factors to weights container
        """
        correction_name = self.case["CMS_eff_id_reco"][self.year][self.id_wp]      
        
        # =============================================================
        #  Muon candidates
        # =============================================================              
        # get muons that pass the id wp, and within SF binning
        muon_pt_mask = (self.m.pt > 15.0) & (self.m.pt < 199.999)
        muon_eta_mask = np.abs(self.m.eta) < 2.39
        muon_id_mask = getattr(self.m, self.muon_map['Id'][self.id_wp])     
        
        in_muon_mask = muon_pt_mask & muon_eta_mask & muon_id_mask
        in_muons = self.m.mask[in_muon_mask]

        # get muons pT and abseta (replace None values with some 'in-limit' value)
        muon_pt = ak.fill_none(in_muons.pt, 15.0)
        muon_eta = np.abs(ak.fill_none(in_muons.eta, 0.0))


        # =============================================================
        # Correction: event-level weight (nominal/up/down)
        # =============================================================         
        # Get nominal, up, and down scale factors
        nominal_sf, up_sf, down_sf = [
            unflat_sf(self.cset[correction_name].evaluate(muon_eta, muon_pt, v), in_muon_mask, self.n)
            for v in ("nominal", "systup", "systdown")
        ]

        nominal_sf, up_sf, down_sf = [
            ak.where(self.muon_mask, sf, 1.0)
            for sf in (nominal_sf, up_sf, down_sf)
        ]

        # add scale factors to weights container
        self.weights.add(
            name=f"CMS_eff_m_id_syst_{self.year}",
            weight=nominal_sf,
            weightUp=up_sf,
            weightDown=down_sf,
        )


    def add_iso_weight(self):
        """
        add muon Iso (LooseRelIso with mediumID) scale factors to weights container
        """
        # Structure: Iso/Id -> RelIso is used as isolation, see wprime_plus_b/json_files/muon.json
        correction_name = self.case["CMS_eff_m_iso"][self.year][f"{self.iso_wp}RelIso"][self.id_wp]
        
        # =============================================================
        #  Muon candidates
        # =============================================================                
        # get 'in-limits' muons
        muon_pt_mask = self.m.pt > 15.0
        muon_eta_mask = np.abs(self.m.eta) < 2.39
        muon_id_mask =  getattr(self.m, self.muon_map['Id'][self.id_wp]) 
        muon_iso_mask = getattr(self.m, self.muon_map['Iso']['Flag']) < self.muon_map['Iso'][self.iso_wp]

        in_muon_mask = muon_pt_mask & muon_eta_mask & muon_id_mask #& muon_iso_mask
        in_muons = self.m.mask[in_muon_mask]

        # get muons pT and abseta (replace None values with some 'in-limit' value)
        muon_pt = ak.fill_none(in_muons.pt, 29.0)
        muon_eta = np.abs(ak.fill_none(in_muons.eta, 0.0))

        # =============================================================
        # Correction: event-level weight (nominal/up/down)
        # =============================================================         
        # Get nominal, up, and down scale factors
        nominal_sf, up_sf, down_sf = [
            unflat_sf(self.cset[correction_name].evaluate(muon_eta, muon_pt, v), in_muon_mask, self.n)
            for v in ("nominal", "systup", "systdown")
        ]
        
        nominal_sf, up_sf, down_sf = [
            ak.where(self.muon_mask, sf, 1.0)
            for sf in (nominal_sf, up_sf, down_sf)
        ]

        # add scale factors to weights container
        self.weights.add(
            name=f"CMS_eff_m_iso_syst_{self.year}",
            weight=nominal_sf,
            weightUp=up_sf,
            weightDown=down_sf,
        )
        

    def add_triggeriso_weight(self, trigger_mask, trigger_match_mask) -> None:
        """
        add muon Trigger Iso (IsoMu24 or IsoMu27) weights
        
        trigger_mask:
            mask array of events passing the analysis trigger
        trigger_match_mask:
            mask array of DeltaR matched trigger objects
        """
        assert (
            self.id_wp == "Tight" and self.iso_wp == "Tight"
        ), "there's only available muon trigger SF for 'tight' ID and Iso"


        correction_name = self.case["CMS_eff_m_trigger"][self.year]
        
        # =============================================================
        #  Muon candidates
        # =============================================================  
        muon_pt_mask = (
            (self.m.pt > 29.0)
        )
        muon_eta_mask = (
            (np.abs(self.m.eta) < 2.4)
        )

        one_muon_per_event = (
            (self.n == 1)
        )

        in_muon_mask = (
            muon_pt_mask & muon_eta_mask
        )        

        in_muons = self.m.mask[in_muon_mask]
        
        # get muons transverse momentum and abs pseudorapidity (replace None values with some 'in-limit' value)
        muon_pt = ak.fill_none(in_muons.pt, 29.0)
        muon_eta = np.abs(ak.fill_none(in_muons.eta, 0.0))

        # =============================================================
        # Correction: event-level weight (nominal/up/down)
        # =============================================================         
        # Get nominal, up, and down scale factors
        nominal_sf_tmp, up_sf_tmp, down_sf_tmp = [
            unflat_sf(self.cset[correction_name].evaluate(muon_eta, muon_pt, v), in_muon_mask, self.n)
            for v in ("nominal", "systup", "systdown")
        ]

        # consider the events that only triggered the trigger.
        event_mask = trigger_mask & trigger_match_mask
        
        nominal_sf, up_sf, down_sf = [
            ak.where(event_mask, sf, 1.0)
            for sf in (nominal_sf_tmp, up_sf_tmp, down_sf_tmp)
        ]

        nominal_sf, up_sf, down_sf = [
            ak.where(self.muon_mask, sf, 1.0)
            for sf in (nominal_sf, up_sf, down_sf)
        ]

        # add scale factors to weights container
        self.weights.add(
            name=f"CMS_eff_m_trigger_{self.year}",
            weight=nominal_sf,
            weightUp=up_sf,
            weightDown=down_sf,
        )
     

    def add_dimuon_trigger_weight(self, trigger_mask, trigger_match_mask) -> None:        
        """
        https://cms.cern.ch/iCMS/jsp/openfile.jsp?tp=draft&files=AN2015_324_v15.pdf

        dilepton events should be computed as
        ε(l1, l2) = 1 - P(both leptons fail)  = 1 - (1 - ε(l1, trig)) · (1 - ε(l2, trig))  = ε(l1, trig) + ε(l2, trig) - ε(l1, trig) · ε(l2, trig)

        ε_data and ε_MC are measured in bins of pT and η

        Thanks Daniel: https://github.com/deoache/bsm3g_coffea/blob/main/notebooks/muon_trigger_eff.ipynb

        """    
        
        correction_name = self.case["CMS_eff_m_trigger"][self.year]

        # =============================================================
        #  Muon candidates
        # =============================================================         
        pt_threshold = {
            "2016APV": 26.0,
            "2016": 26.0,
            "2017": 29.0,
            "2018": 29.0,
        }
        # get 'in-limits' muons
        muon_pt_mask = (
            (self.m.pt > pt_threshold[self.year]) 
            & (self.m.pt < 199.99) 
        )
        muon_eta_mask = (np.abs(self.m.eta) < 2.399)

        in_muon_mask = muon_pt_mask & muon_eta_mask  

        in_muons = self.m.mask[in_muon_mask]
            
        muon_pt = ak.fill_none(in_muons.pt, 30.0)
        muon_eta = np.abs(ak.fill_none(in_muons.eta, 0.0))

        # =============================================================
        # Correction: event-level weight (nominal/up/down)
        # =============================================================         
        # Calculate SF
        double_cset = correctionlib.CorrectionSet.from_file(
                f"{Path.cwd()}/wprime_plus_b/corrections/HLT/mu/{self.year}_Muon_HLT_Eff.json"
        )

        data_eff = double_cset["Muon-HLT-DataEff"].evaluate(
                self.variation,
                correction_name,
                muon_eta,
                muon_pt,
        )

        mc_eff = double_cset["Muon-HLT-McEff"].evaluate(
                self.variation,
                correction_name,
                muon_eta,
                muon_pt,
        )

        # Removing arbitrary values
        data_eff = ak.where(in_muon_mask, data_eff, ak.ones_like(data_eff))
        mc_eff = ak.where(in_muon_mask, mc_eff, ak.ones_like(mc_eff))

        # Unflattening the arrays
        data_eff = ak.unflatten(data_eff, self.n)
        mc_eff = ak.unflatten(mc_eff, self.n)


        # Obtening the first and second elements of the arrays
        data_eff_1 = ak.firsts(data_eff)
        data_eff_2 = ak.pad_none(data_eff, target=2)[:, 1]

        
        mc_eff_1 = ak.firsts(mc_eff)
        mc_eff_2 = ak.pad_none(mc_eff, target=2)[:, 1]        

        # Calculating the full efficiency
        full_data_eff = data_eff_1 + data_eff_2 - data_eff_1 * data_eff_2
        full_mc_eff = mc_eff_1 + mc_eff_2 - mc_eff_1 * mc_eff_2

        nominal_sf_tmp = full_data_eff / full_mc_eff


        # consider the events that only triggered the trigger.
        event_mask = trigger_mask & trigger_match_mask    
        nominal_sf = ak.where(event_mask, nominal_sf_tmp, 1.0)
         
        nominal_sf, up_sf, down_sf = [
            ak.where(self.muon_mask, sf, 1.0)
            for sf in (nominal_sf, up_sf, down_sf)
        ]

        self.weights.add(
            name=f"CMS_eff_m_trigger_{self.year}",
            weight=nominal_sf,
        )