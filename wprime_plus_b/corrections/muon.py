import json
import copy
import correctionlib
import numpy as np
import awkward as ak
import importlib.resources
from typing import Type
from pathlib import Path
from .utils import unflat_sf
from coffea.analysis_tools import Weights
from wprime_plus_b.corrections.utils import pog_years, get_pog_json

# https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2016
# https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2017
# https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2018

def get_id_wps(muons):
    return {
        # cutbased ID working points
        "Loose": muons.looseId,
        "Medium": muons.mediumId,
        "Tight": muons.tightId,
    }

def get_iso_wps(muons):
    return {
        "Loose": (
            muons.pfRelIso04_all < 0.25
            if hasattr(muons, "pfRelIso04_all")
            else muons.pfRelIso03_all < 0.25
        ),
        "Medium": (
            muons.pfRelIso04_all < 0.20
            if hasattr(muons, "pfRelIso04_all")
            else muons.pfRelIso03_all < 0.20
        ),
        "Tight": (
            muons.pfRelIso04_all < 0.15
            if hasattr(muons, "pfRelIso04_all")
            else muons.pfRelIso03_all < 0.15
        ),
    }


class MuonCorrector:
    """
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
        variation: str = "nominal",
        id_wp: str = "Tight",
        iso_wp: str = "Tight",
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
        self.weights = weights
        
        # define correction set
        self.cset = correctionlib.CorrectionSet.from_file(
            get_pog_json(json_name="muon", year=year)
        )
        self.year = year
        self.pog_year = pog_years[year]

    def add_reco_weight(self):
        """
        add muon RECO scale factors to weights container
        """
        # get muons within SF binning
        muon_pt_mask = self.m.pt >= 40.0
        muon_eta_mask = np.abs(self.m.eta) < 2.4
        in_muon_mask = muon_pt_mask & muon_eta_mask 
        in_muons = self.m.mask[in_muon_mask]

        # get muons pT and abseta (replace None values with some 'in-limit' value)
        muon_pt = ak.fill_none(in_muons.pt, 40.0)
        muon_eta = np.abs(ak.fill_none(in_muons.eta, 0.0))
        
        # 'id' scale factors names
        reco_corrections = {
            "2016APV": "NUM_TrackerMuons_DEN_genTracks", 
            "2016": "NUM_TrackerMuons_DEN_genTracks",
            "2017": "NUM_TrackerMuons_DEN_genTracks",
            "2018": "NUM_TrackerMuons_DEN_genTracks",
        }
        # get nominal scale factors
        nominal_sf = unflat_sf(
            self.cset[reco_corrections[self.year]].evaluate(
                muon_eta, muon_pt, "nominal"
            ),
            in_muon_mask,
            self.n,
        )
        # get 'up' and 'down' scale factors
        up_sf = unflat_sf(
            self.cset[reco_corrections[self.year]].evaluate(
                muon_eta, muon_pt, "systup"
            ),
            in_muon_mask,
            self.n,
        )
        down_sf = unflat_sf(
            self.cset[reco_corrections[self.year]].evaluate(
                muon_eta, muon_pt, "systdown"
            ),
            in_muon_mask,
            self.n,
        )
        # add scale factors to weights container
        self.weights.add(
            name=f"CMS_eff_m_reco_syst",
            weight=nominal_sf,
            weightUp=up_sf,
            weightDown=down_sf,
        )
            

    def add_id_weight(self):
        """
        add muon ID scale factors to weights container
        """
        # get muons that pass the id wp, and within SF binning
        muon_pt_mask = (self.m.pt > 15.0) & (self.m.pt < 199.999)
        muon_eta_mask = np.abs(self.m.eta) < 2.39
        muon_id_mask = get_id_wps(self.m)[self.id_wp]
        in_muon_mask = muon_pt_mask & muon_eta_mask & muon_id_mask
        in_muons = self.m.mask[in_muon_mask]

        # get muons pT and abseta (replace None values with some 'in-limit' value)
        muon_pt = ak.fill_none(in_muons.pt, 15.0)
        muon_eta = np.abs(ak.fill_none(in_muons.eta, 0.0))

        # 'id' scale factors names
        id_corrections = {
            "2016APV": {
                "Loose": "NUM_LooseID_DEN_TrackerMuons",
                "Medium": "NUM_MediumID_DEN_TrackerMuons",
                "Tight": "NUM_TightID_DEN_TrackerMuons",
            },
            "2016": {
                "Loose": "NUM_LooseID_DEN_TrackerMuons",
                "Medium": "NUM_MediumID_DEN_TrackerMuons",
                "Tight": "NUM_TightID_DEN_TrackerMuons",
            },
            "2017": {
                "Loose": "NUM_LooseID_DEN_TrackerMuons",
                "Medium": "NUM_MediumID_DEN_TrackerMuons",
                "Tight": "NUM_TightID_DEN_TrackerMuons",
            },
            "2018": {
                "Loose": "NUM_LooseID_DEN_TrackerMuons",
                "Medium": "NUM_MediumID_DEN_TrackerMuons",
                "Tight": "NUM_TightID_DEN_TrackerMuons",
            },
        }

        # get nominal scale factors
        nominal_sf = unflat_sf(
            self.cset[id_corrections[self.year][self.id_wp]].evaluate(
                muon_eta, muon_pt, "nominal"
            ),
            in_muon_mask,
            self.n,
        )
        # get 'up' and 'down' scale factors
        up_sf = unflat_sf(
            self.cset[
                id_corrections[self.year][self.id_wp]
            ].evaluate(muon_eta, muon_pt, "systup"),
            in_muon_mask,
            self.n,
        )
        down_sf = unflat_sf(
            self.cset[
                id_corrections[self.year][self.id_wp]
            ].evaluate(muon_eta, muon_pt, "systdown"),
            in_muon_mask,
            self.n,
        )
        # add scale factors to weights container
        self.weights.add(
            name=f"CMS_eff_m_id_syst",
            weight=nominal_sf,
            weightUp=up_sf,
            weightDown=down_sf,
        )


    def add_iso_weight(self):
        """
        add muon Iso (LooseRelIso with mediumID) scale factors to weights container
        """
        # get 'in-limits' muons
        muon_pt_mask = self.m.pt > 15.0
        muon_eta_mask = np.abs(self.m.eta) < 2.39
        muon_id_mask = get_id_wps(self.m)[self.id_wp]
        muon_iso_mask = get_iso_wps(self.m)[self.iso_wp]
        in_muon_mask = muon_pt_mask & muon_eta_mask & muon_id_mask & muon_iso_mask
        in_muons = self.m.mask[in_muon_mask]

        # get muons pT and abseta (replace None values with some 'in-limit' value)
        muon_pt = ak.fill_none(in_muons.pt, 29.0)
        muon_eta = np.abs(ak.fill_none(in_muons.eta, 0.0))

        iso_corrections = {
            "Run_2": {
                "Loose": {
                    "Loose": "NUM_LooseRelIso_DEN_LooseID",
                    "Medium": None,
                    "Tight": None,
                },
                "Medium": {
                    "Loose": "NUM_LooseRelIso_DEN_MediumID",
                    "Medium": None,
                    "Tight": "NUM_TightRelIso_DEN_MediumID",
                },
                "Tight": {
                    "Loose": "NUM_LooseRelIso_DEN_TightIDandIPCut",
                    "Medium": None,
                    "Tight": "NUM_TightRelIso_DEN_TightIDandIPCut",
                },
            },
            "Run_3": {
                "Loose": {
                    "Loose": "NUM_LoosePFIso_DEN_LooseID",
                    "Medium": "NUM_LoosePFIso_DEN_MediumID",
                    "Tight": "NUM_LoosePFIso_DEN_TightID",
                },
                "Medium": {
                    "Loose": None,
                    "Medium": None,
                    "Tight": None,
                },
                "Tight": {
                    "Loose": None,
                    "Medium": "NUM_TightPFIso_DEN_MediumID",
                    "Tight": "NUM_TightPFIso_DEN_TightID",
                },
            }
        }

        run_case = "Run_2" if self.year in ["2016APV", "2016", "2017", "2018"] else "Run_3"

        correction_name = iso_corrections[run_case][self.id_wp][self.iso_wp]
        assert correction_name, "No Iso SF's available"

        # get nominal scale factors
        nominal_sf = unflat_sf(
            self.cset[correction_name].evaluate(muon_eta, muon_pt, "nominal"),
            in_muon_mask,
            self.n,
        )
        # get 'up' and 'down' scale factors
        up_sf = unflat_sf(
            self.cset[correction_name].evaluate(
                muon_eta, muon_pt, "systup"
            ),
            in_muon_mask,
            self.n,
        )
        down_sf = unflat_sf(
            self.cset[correction_name].evaluate(
                muon_eta, muon_pt, "systdown"
            ),
            in_muon_mask,
            self.n,
        )
        # add scale factors to weights container
        self.weights.add(
            name=f"CMS_eff_m_iso_syst",
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
        
        # get 'in-limits' muons
        muon_pt_mask = (self.m.pt > 29.0) #& (self.m.pt < 199.999)
        muon_eta_mask = np.abs(self.m.eta) < 2.399
        muon_id_mask = get_id_wps(self.m)[self.id_wp]
        muon_iso_mask = get_iso_wps(self.m)[self.iso_wp]
        
        trigger_mask = ak.flatten(ak.ones_like(self.muons.pt) * trigger_mask) > 0
        trigger_match_mask = ak.flatten(trigger_match_mask)
        

        in_muon_mask = (
            muon_pt_mask & muon_eta_mask & muon_id_mask & muon_iso_mask & trigger_mask & trigger_match_mask
        )
        in_muons = self.m.mask[in_muon_mask]

        # get muons transverse momentum and abs pseudorapidity (replace None values with some 'in-limit' value)
        muon_pt = ak.fill_none(in_muons.pt, 29.0)
        muon_eta = np.abs(ak.fill_none(in_muons.eta, 0.0))

        # scale factors keys
        sfs_keys = {
            "2016APV": "NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight",
            "2016": "NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight",
            "2017": "NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight",
            "2018": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",
        }
        # get nominal scale factors
        sf = self.cset[sfs_keys[self.year]].evaluate(
            muon_eta, muon_pt, "nominal"
        )
        nominal_sf = unflat_sf(
            sf,
            in_muon_mask,
            self.n,
        )
        # get 'up' and 'down' scale factors
        up_sf = self.cset[sfs_keys[self.year]].evaluate(
            muon_eta, muon_pt, "systup"
        )
        up_sf = unflat_sf(
            up_sf,
            in_muon_mask,
            self.n,
        )
        down_sf = self.cset[sfs_keys[self.year]].evaluate(
            muon_eta, muon_pt, "systdown"
        )
        down_sf = unflat_sf(
            down_sf,
            in_muon_mask,
            self.n,
        )
        # add scale factors to weights container
        self.weights.add(
            name=f"muon_triggeriso",
            weight=nominal_sf,
            weightUp=up_sf,
            weightDown=down_sf,
        )
     

    def add_dimuon_trigger_weight(self) -> None:        
        """
        https://cms.cern.ch/iCMS/jsp/openfile.jsp?tp=draft&files=AN2015_324_v15.pdf

        dilepton events should be computed as
        ε(l1, l2) = 1 - P(both leptons fail)  = 1 - (1 - ε(l1, trig)) · (1 - ε(l2, trig))  = ε(l1, trig) + ε(l2, trig) - ε(l1, trig) · ε(l2, trig)

        ε_data and ε_MC are measured in bins of pT and η

        Thanks Daniel: https://github.com/deoache/bsm3g_coffea/blob/main/notebooks/muon_trigger_eff.ipynb

        """    
        
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

        sfs_keys = {
            "2016APV": "NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight",
            "2016": "NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight",
            "2017": "NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight",
            "2018": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",
        }


        double_cset = correctionlib.CorrectionSet.from_file(
                f"{Path.cwd()}/wprime_plus_b/data/{self.year}_Muon_HLT_Eff.json"
        )

        data_eff = double_cset["Muon-HLT-DataEff"].evaluate(
                self.variation,
                sfs_keys[self.year],
                muon_eta,
                muon_pt,
        )

        mc_eff = double_cset["Muon-HLT-McEff"].evaluate(
                self.variation,
                sfs_keys[self.year],
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

        nominal_sf = full_data_eff / full_mc_eff

        self.weights.add(
                    name=f"dimuon_trigger_{self.year}",
                    weight=nominal_sf,
        )