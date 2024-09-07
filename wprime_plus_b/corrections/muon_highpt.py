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


def get_iso_wps(muons):
    return {
        "loose": (
            muons.pfRelIso04_all < 0.25
            if hasattr(muons, "pfRelIso04_all")
            else muons.pfRelIso03_all < 0.25
        ),
        "medium": (
            muons.pfRelIso04_all < 0.20
            if hasattr(muons, "pfRelIso04_all")
            else muons.pfRelIso03_all < 0.20
        ),
        "tight": (
            muons.pfRelIso04_all < 0.15
            if hasattr(muons, "pfRelIso04_all")
            else muons.pfRelIso03_all < 0.15
        ),
    }


class MuonHighPtCorrector:
    """
    Muon corrector class

    Parameters:
    -----------
    muons:
        muons collection
    weights:
        Weights object from coffea.analysis_tools
    year:
        Year of the dataset {'2016', '2016APV', '2017', '2018'}
    variation:
        syst variation
    id_wp:
        ID working point {'highpt'}
    iso_wp:
        Iso working point {'loose', 'medium', 'tight'}
    """

    def __init__(
        self,
        muons: ak.Array,
        weights: Type[Weights],
        year: str = "2017",
        variation: str = "nominal",
        id_wp: str = "highpt",
        iso_wp: str = "tight",
    ) -> None:
        self.muons = muons
        self.variation = variation
        self.iso_wp = iso_wp
        self.id_wp = id_wp
        self.year = year
        self.pog_year = pog_years[year]

        # muon array
        self.muons = muons
        # flat muon array
        self.m, self.n = ak.flatten(muons), ak.num(muons)
        # weights container
        self.weights = weights
        # define correction set
        self.cset = correctionlib.CorrectionSet.from_file(
            get_pog_json(json_name="muon_highpt", year=year)
        )
        
    def add_reco_weight(self):
        """
        add muon RECO scale factors to weights container
        """
        # get muons within SF binning
        muon_pt_mask = self.m.pt >= 50.0
        muon_eta_mask = np.abs(self.m.eta) < 2.4
        in_muon_mask = muon_pt_mask & muon_eta_mask 
        in_muons = self.m.mask[in_muon_mask]

        # get muons pT and abseta (replace None values with some 'in-limit' value)
        muon_pt = ak.fill_none(in_muons.pt, 50.0)
        muon_eta = np.abs(ak.fill_none(in_muons.eta, 0.0))

        # 'id' scale factors names
        reco_corrections = {
            "2016APV": "NUM_GlobalMuons_DEN_TrackerMuonProbes", 
            "2016": "NUM_GlobalMuons_DEN_TrackerMuonProbes",
            "2017": "NUM_GlobalMuons_DEN_TrackerMuonProbes",
            "2018": "NUM_GlobalMuons_DEN_TrackerMuonProbes",
        }
        # get nominal scale factors
        nominal_sf = unflat_sf(
            self.cset[reco_corrections[self.year]].evaluate(
                muon_eta, muon_pt, "nominal"
            ),
            in_muon_mask,
            self.n,
        )
        if self.variation == "nominal":
            # get 'up' and 'down' scale factors
            up_sf = unflat_sf(
                self.cset[
                    reco_corrections[self.year]].evaluate(
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
                name=f"muon_highpt_reco",
                weight=nominal_sf,
                weightUp=up_sf,
                weightDown=down_sf,
            )
        else:
            self.weights.add(
                name=f"muon_highpt_reco",
                weight=nominal_sf,
            )

    def add_id_weight(self):
        """
        add muon ID scale factors to weights container
        """
        # get muons that pass the id wp, and within SF binning
        muon_pt_mask = self.m.pt > 50.0
        muon_eta_mask = np.abs(self.m.eta) < 2.39
        muon_id_mask = self.m.highPtId == 2
        in_muon_mask = muon_pt_mask & muon_eta_mask & muon_id_mask
        in_muons = self.m.mask[in_muon_mask]

        # get muons pT and abseta (replace None values with some 'in-limit' value)
        muon_pt = ak.fill_none(in_muons.pt, 50.0)
        muon_eta = np.abs(ak.fill_none(in_muons.eta, 0.0))

        # 'id' scale factors names
        id_corrections = {
            "2016APV": {"highpt": "NUM_HighPtID_DEN_GlobalMuonProbes"},
            "2016": {"highpt": "NUM_HighPtID_DEN_GlobalMuonProbes"},
            "2017": {"highpt": "NUM_HighPtID_DEN_GlobalMuonProbes"},
            "2018": {"highpt": "NUM_HighPtID_DEN_GlobalMuonProbes"},
        }

        # get nominal scale factors
        nominal_sf = unflat_sf(
            self.cset[id_corrections[self.year][self.id_wp]].evaluate(
                muon_eta, muon_pt, "nominal"
            ),
            in_muon_mask,
            self.n,
        )
        if self.variation == "nominal":
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
                name=f"muon_highpt_id",
                weight=nominal_sf,
                weightUp=up_sf,
                weightDown=down_sf,
            )
        else:
            self.weights.add(
                name=f"muon_highpt_id",
                weight=nominal_sf,
            )

    def add_iso_weight(self):
        """
        add muon Iso scale factors to weights container
        """
        # get 'in-limits' muons
        muon_pt_mask = self.m.pt > 50.0
        muon_eta_mask = np.abs(self.m.eta) < 2.39
        muon_id_mask = self.m.highPtId == 2
        muon_iso_mask = get_iso_wps(self.m)[self.iso_wp]
        in_muon_mask = muon_pt_mask & muon_eta_mask & muon_id_mask & muon_iso_mask
        in_muons = self.m.mask[in_muon_mask]

        # get muons pT and abseta (replace None values with some 'in-limit' value)
        muon_pt = ak.fill_none(in_muons.pt, 50.0)
        muon_eta = np.abs(ak.fill_none(in_muons.eta, 0.0))

        iso_corrections = {
            "2016APV": {
                "loose": "NUM_probe_LooseRelTkIso_DEN_HighPtProbes",
                "medium": None,
                "tight": "NUM_probe_TightRelTkIso_DEN_HighPtProbes",
            },
            "2016": {
                "loose": "NUM_probe_LooseRelTkIso_DEN_HighPtProbes",
                "medium": None,
                "tight": "NUM_probe_TightRelTkIso_DEN_HighPtProbes",
            },
            "2017": {
                "loose": "NUM_probe_LooseRelTkIso_DEN_HighPtProbes",
                "medium": None,
                "tight": "NUM_probe_TightRelTkIso_DEN_HighPtProbes",
            },
            "2018": {
                "loose": "NUM_probe_LooseRelTkIso_DEN_HighPtProbes",
                "medium": None,
                "tight": "NUM_probe_TightRelTkIso_DEN_HighPtProbes",
            },
        }

        correction_name = iso_corrections[self.year][self.iso_wp]
        assert correction_name, "No Iso SF's available"

        # get nominal scale factors
        nominal_sf = unflat_sf(
            self.cset[correction_name].evaluate(muon_eta, muon_pt, "nominal"),
            in_muon_mask,
            self.n,
        )
        if self.variation == "nominal":
            # get 'up' and 'down' scale factors
            up_sf = unflat_sf(
                self.cset[correction_name].evaluate(muon_eta, muon_pt, "systup"),
                in_muon_mask,
                self.n,
            )
            down_sf = unflat_sf(
                self.cset[correction_name].evaluate(muon_eta, muon_pt, "systdown"),
                in_muon_mask,
                self.n,
            )
            # add scale factors to weights container
            self.weights.add(
                name=f"muon_highpt_iso",
                weight=nominal_sf,
                weightUp=up_sf,
                weightDown=down_sf,
            )
        else:
            self.weights.add(
                name=f"muon_highpt_iso",
                weight=nominal_sf,
            )

    def add_triggeriso_weight(self, trigger_mask, trigger_match_mask) -> None:
        """
        add muon Trigger Iso scale factors.

        trigger_mask:
            mask array of events passing the analysis trigger
        trigger_match_mask:
            mask array of DeltaR matched trigger objects
        """
        # get 'in-limits' muons
        muon_pt_mask = self.m.pt > 50.0
        muon_eta_mask = np.abs(self.m.eta) < 2.399
        muon_id_mask = self.m.highPtId == 2
        muon_iso_mask = get_iso_wps(self.m)[self.iso_wp]

        trigger_mask = ak.flatten(ak.ones_like(self.muons.pt) * trigger_mask) > 0
        trigger_match_mask = ak.flatten(trigger_match_mask)

        in_muon_mask = (
            muon_pt_mask
            & muon_eta_mask
            & muon_id_mask
            & muon_iso_mask
            & trigger_mask
            & trigger_match_mask
        )
        in_muons = self.m.mask[in_muon_mask]

        # get muons transverse momentum and abs pseudorapidity (replace None values with some 'in-limit' value)
        muon_pt = ak.fill_none(in_muons.pt, 50.0)
        muon_eta = np.abs(ak.fill_none(in_muons.eta, 0.0))

        # scale factors keys
        sfs_keys = {
            "2016APV": "NUM_HLT_DEN_HighPtTightRelIsoProbes",
            "2016": "NUM_HLT_DEN_HighPtTightRelIsoProbes",
            "2017": "NUM_HLT_DEN_HighPtTightRelIsoProbes",
            "2018": "NUM_HLT_DEN_HighPtTightRelIsoProbes",
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
        if self.variation == "nominal":
            # get 'up' and 'down' scale factors
            up_sf = self.cset[sfs_keys[self.year]].evaluate(muon_eta, muon_pt, "systup")
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
                name=f"muon_highpt_triggeriso",
                weight=nominal_sf,
                weightUp=up_sf,
                weightDown=down_sf,
            )
        else:
            self.weights.add(
                name=f"muon_highpt_triggeriso",
                weight=nominal_sf,
            )