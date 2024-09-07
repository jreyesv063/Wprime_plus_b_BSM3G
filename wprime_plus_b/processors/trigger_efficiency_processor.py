import json
import hist
import pickle
import numpy as np
import pandas as pd
import awkward as ak
from typing import List
from coffea import processor
from coffea.analysis_tools import Weights, PackedSelection
from wprime_plus_b.processors.utils.analysis_utils import delta_r_mask, normalize
from wprime_plus_b.corrections.jec import jet_corrections
from wprime_plus_b.corrections.met import met_phi_corrections
from wprime_plus_b.corrections.btag import BTagCorrector
from wprime_plus_b.corrections.pileup import add_pileup_weight
from wprime_plus_b.corrections.l1prefiring import add_l1prefiring_weight
from wprime_plus_b.corrections.pujetid import add_pujetid_weight
from wprime_plus_b.corrections.lepton import (
    ElectronCorrector,
    MuonCorrector,
    TauCorrector,
)
from wprime_plus_b.corrections.tau_energy import tau_energy_scale, met_corrected_tes


class TriggerEfficiencyProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year: str = "2017",
        yearmod: str = "",
        lepton_flavor: str = "ele",
        output_type: str = "hist",
    ):
        self._year = year
        self._yearmod = yearmod
        self._lepton_flavor = lepton_flavor

        # open triggers
        with open("wprime_plus_b/data/triggers.json", "r") as f:
            self._triggers = json.load(f)[self._year]
        # open btagDeepFlavB
        with open("wprime_plus_b/data/btagDeepFlavB.json", "r") as f:
            self._btagDeepFlavB = json.load(f)[self._year]
        # open met filters
        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        with open("wprime_plus_b/data/metfilters.json", "rb") as handle:
            self._metfilters = json.load(handle)[self._year]
        # open lumi masks
        with open("wprime_plus_b/data/lumi_masks.pkl", "rb") as handle:
            self._lumi_mask = pickle.load(handle)
        # output histograms
        self.make_output = lambda: {
            "electron_kin": hist.Hist(
                hist.axis.StrCategory([], name="region", growth=True),
                hist.axis.Variable(
                    [30, 60, 90, 120, 150, 180, 210, 240, 300, 500],
                    name="electron_pt",
                    label=r"electron $p_T$ [GeV]",
                ),
                hist.axis.Regular(
                    50, -2.4, 2.4, name="electron_eta", label="electron $\eta$"
                ),
                hist.storage.Weight(),
            ),
            "muon_kin": hist.Hist(
                hist.axis.StrCategory([], name="region", growth=True),
                hist.axis.Variable(
                    [30, 60, 90, 120, 150, 180, 210, 240, 300, 500],
                    name="muon_pt",
                    label=r"muon $p_T$ [GeV]",
                ),
                hist.axis.Regular(50, -2.4, 2.4, name="muon_eta", label="muon $\eta$"),
                hist.storage.Weight(),
            ),
            "jet_kin": hist.Hist(
                hist.axis.StrCategory([], name="region", growth=True),
                hist.axis.Variable(
                    [30, 60, 90, 120, 150, 180, 210, 240, 300, 500],
                    name="jet_pt",
                    label=r"bJet $p_T$ [GeV]",
                ),
                hist.axis.Regular(50, -2.4, 2.4, name="jet_eta", label="bJet $\eta$"),
                hist.storage.Weight(),
            ),
            "met_kin": hist.Hist(
                hist.axis.StrCategory([], name="region", growth=True),
                hist.axis.Variable(
                    [50, 75, 100, 125, 150, 175, 200, 300, 500],
                    name="met_pt",
                    label=r"$p_T^{miss}$ [GeV]",
                ),
                hist.axis.Regular(
                    50, -4.0, 4.0, name="met_phi", label=r"$\phi(p_T^{miss})$"
                ),
                hist.storage.Weight(),
            ),
            "electron_bjet_kin": hist.Hist(
                hist.axis.StrCategory([], name="region", growth=True),
                hist.axis.Regular(
                    30, 0, 5, name="electron_bjet_dr", label="$\Delta R(e, bJet)$"
                ),
                hist.axis.Variable(
                    [40, 75, 100, 125, 150, 175, 200, 300, 500],
                    name="invariant_mass",
                    label=r"$m(e, bJet)$ [GeV]",
                ),
                hist.storage.Weight(),
            ),
            "muon_bjet_kin": hist.Hist(
                hist.axis.StrCategory([], name="region", growth=True),
                hist.axis.Regular(
                    30, 0, 5, name="muon_bjet_dr", label="$\Delta R(\mu, bJet)$"
                ),
                hist.axis.Variable(
                    [40, 75, 100, 125, 150, 175, 200, 300, 500],
                    name="invariant_mass",
                    label=r"$m(\mu, bJet)$ [GeV]",
                ),
                hist.storage.Weight(),
            ),
            "lep_met_kin": hist.Hist(
                hist.axis.StrCategory([], name="region", growth=True),
                hist.axis.Variable(
                    [40, 75, 100, 125, 150, 175, 200, 300, 500, 800],
                    name="electron_met_transverse_mass",
                    label=r"$m_T(e, p_T^{miss})$ [GeV]",
                ),
                hist.axis.Variable(
                    [40, 75, 100, 125, 150, 175, 200, 300, 500, 800],
                    name="muon_met_transverse_mass",
                    label=r"$m_T(\mu, p_T^{miss})$ [GeV]",
                ),
                hist.storage.Weight(),
            ),
            "lep_bjet_met_kin": hist.Hist(
                hist.axis.StrCategory([], name="region", growth=True),
                hist.axis.Variable(
                    [40, 75, 100, 125, 150, 175, 200, 300, 500, 800],
                    name="electron_total_transverse_mass",
                    label=r"$m_T^{tot}(e, bJet, p_T^{miss})$ [GeV]",
                ),
                hist.axis.Variable(
                    [40, 75, 100, 125, 150, 175, 200, 300, 500, 800],
                    name="muon_total_transverse_mass",
                    label=r"$m_T^{tot}(\mu, bJet, p_T^{miss})$ [GeV]",
                ),
                hist.storage.Weight(),
            ),
        }

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        dataset = events.metadata["dataset"]
        nevents = len(events)
        self.is_mc = hasattr(events, "genWeight")
        self.histograms = self.make_output()

        # dictionary to store output data and metadata
        output = {}

        # get triggers masks
        trigger_mask = {}
        for ch in ["ele", "mu"]:
            trigger_mask[ch] = np.zeros(nevents, dtype="bool")
            for t in self._triggers[ch]:
                if t in events.HLT.fields:
                    trigger_mask[ch] = trigger_mask[ch] | events.HLT[t]
                    
        # apply corrections to jet/met
        if self.is_mc:
            corrected_jets, met = jet_corrections(events, self._year)
        else:
            corrected_jets, met = events.Jet, events.MET
            
        # --------------------
        # object selection
        # --------------------
        # select electrons
        good_electrons = (
            (events.Electron.pt >= 30)
            & (np.abs(events.Electron.eta) < 2.4)
            & (
                (np.abs(events.Electron.eta) < 1.44)
                | (np.abs(events.Electron.eta) > 1.57)
            )
            & (
                events.Electron.mvaFall17V2Iso_WP80
                if self._lepton_flavor == "ele"
                else events.Electron.mvaFall17V2Iso_WP90
            )
        )
        electrons = events.Electron[good_electrons]  
        # select muons
        good_muons = (
            (events.Muon.pt >= 30)
            & (np.abs(events.Muon.eta) < 2.4)
            & (events.Muon.tightId)
            & (
                events.Muon.pfRelIso04_all < 0.15
                if hasattr(events.Muon, "pfRelIso04_all")
                else events.Muon.pfRelIso03_all < 0.15
            )
        )
        good_muons = (good_muons) & (
            delta_r_mask(events.Muon, electrons, threshold=0.4)
        )
        muons = events.Muon[good_muons]
        # correct and select muons
        # apply Tau energy corrections (only to MC)
        corrected_taus = events.Tau
        if self.is_mc:
            # Data does not have corrections
            corrected_taus["pt"], corrected_taus["mass"] = tau_energy_scale(
                events, "2017", "", "DeepTau2017v2p1", "nom"
            )
            # Given the tau corrections. We need to recalculate the MET.
            # https://github.com/columnflow/columnflow/blob/16d35bb2f25f62f9110a8f1089e8dc5c62b29825/columnflow/calibration/util.py#L42
            # https://github.com/Katsch21/hh2bbtautau/blob/e268752454a0ce0089ff08cc6c373a353be77679/hbt/calibration/tau.py#L117

            met["pt"], met["phi"] = met_corrected_tes(
                old_taus=events.Tau, new_taus=corrected_taus, met=met
            )
        tau_dm = corrected_taus.decayMode
        decay_mode_mask = ak.zeros_like(tau_dm)
        for mode in [0, 1, 2, 10, 11]:
            decay_mode_mask = np.logical_or(decay_mode_mask, tau_dm == mode)
        good_taus = (
            (corrected_taus.pt > 20)
            & (np.abs(corrected_taus.eta) < 2.3)
            & (np.abs(corrected_taus.dz) < 0.2)
            & (corrected_taus.idDeepTau2017v2p1VSjet > 32)
            & (corrected_taus.idDeepTau2017v2p1VSe > 32)
            & (corrected_taus.idDeepTau2017v2p1VSmu > 8)
            & (decay_mode_mask)
        )
        good_taus = (
            (good_taus)
            & (delta_r_mask(events.Tau, electrons, threshold=0.4))
            & (delta_r_mask(events.Tau, muons, threshold=0.4))
        )
        taus = corrected_taus[good_taus]
        # b-jets
        # break up selection for low and high pT jets
        low_pt_jets_mask = (
            (corrected_jets.pt > 20)
            & (corrected_jets.pt < 50)
            & (np.abs(corrected_jets.eta) < 2.4)
            & (corrected_jets.jetId == 6)
            & (corrected_jets.puId == 7)
            & (corrected_jets.btagDeepFlavB > self._btagDeepFlavB)
        )
        high_pt_jets_mask = (
            (corrected_jets.pt >= 50)
            & (np.abs(corrected_jets.eta) < 2.4)
            & (corrected_jets.jetId == 6)
            & (corrected_jets.btagDeepFlavB > self._btagDeepFlavB)
        )
        good_bjets = ak.where(
            (corrected_jets.pt > 20) & (corrected_jets.pt < 50),
            low_pt_jets_mask,
            high_pt_jets_mask,
        )
        good_bjets = (
            good_bjets
            & (delta_r_mask(corrected_jets, electrons, threshold=0.4))
            & (delta_r_mask(corrected_jets, muons, threshold=0.4))
        )
        bjets = corrected_jets[good_bjets]

        # apply MET phi corrections
        met_pt, met_phi = met_phi_corrections(
            met_pt=met.pt,
            met_phi=met.phi,
            npvs=events.PV.npvsGood,
            run=events.run,
            is_mc=self.is_mc,
            year=self._year,
            year_mod="",
        )
        met["pt"], met["phi"] = met_pt, met_phi

        # --------------------
        # event weights vector
        # --------------------
        weights_container = Weights(len(events), storeIndividual=True)
        if self.is_mc:
            # add gen weigths
            weights_container.add("genweight", events.genWeight)
            # add l1prefiring weigths
            add_l1prefiring_weight(events, weights_container, self._year, "nominal")
            # add pileup weigths
            add_pileup_weight(events, weights_container, self._year, "", "nominal")
            # add pujetid weigths
            add_pujetid_weight(
                jets=corrected_jets,
                weights=weights_container,
                year=self._year,
                year_mod="",
                working_point="M",
                variation="nominal",
            )
            # b-tagging corrector
            btag_corrector = BTagCorrector(
                jets=corrected_jets,
                weights=weights_container,
                sf_type="comb",
                worging_point="M",
                tagger="deepJet",
                year=self._year,
                year_mod="",
                full_run=False,
                variation="nominal",
            )
            # add b-tagging weights
            btag_corrector.add_btag_weights(flavor="bc")
            # electron corrector
            electron_corrector = ElectronCorrector(
                electrons=events.Electron,
                weights=weights_container,
                year=self._year,
                year_mod="",
                variation="nominal",
            )
            # add electron ID weights
            electron_corrector.add_id_weight(
                id_working_point=(
                    "wp80iso" if self._lepton_flavor == "ele" else "wp90iso"
                )
            )
            # add electron reco weights
            electron_corrector.add_reco_weight()

            # muon corrector
            muon_corrector = MuonCorrector(
                muons=events.Muon,
                weights=weights_container,
                year=self._year,
                year_mod="",
                variation="nominal",
                id_wp="tight",
                iso_wp="tight",
            )
            # add muon ID weights
            muon_corrector.add_id_weight()
            # add muon iso weights
            muon_corrector.add_iso_weight()

            # add trigger weights
            if self._lepton_flavor == "ele":
                muon_corrector.add_triggeriso_weight(trigger_mask=trigger_mask["mu"])
                
            # tau corrections
            tau_corrector = TauCorrector(
                taus=corrected_taus,
                weights=weights_container,
                year=self._year,
                year_mod=self._yearmod,
                tau_vs_jet="Tight",
                tau_vs_ele="Tight",
                tau_vs_mu="Tight",
                variation="nominal",
            )
            tau_corrector.add_id_weight_DeepTau2017v2p1VSe()
            tau_corrector.add_id_weight_DeepTau2017v2p1VSmu()
            tau_corrector.add_id_weight_DeepTau2017v2p1VSjet()
            
        # save sum of weights before selections
        output["metadata"] = {"sumw": ak.sum(weights_container.weight())}

        # ---------------
        # event selection
        # ---------------
        # make a PackedSelection object to store selection masks
        self.selections = PackedSelection()

        # luminosity
        if not self.is_mc:
            lumi_mask = self._lumi_mask[self._year](events.run, events.luminosityBlock)
        else:
            lumi_mask = np.ones(len(events), dtype="bool")
        self.selections.add("lumi", lumi_mask)

        # MET filters
        metfilters = np.ones(nevents, dtype="bool")
        metfilterkey = "mc" if self.is_mc else "data"
        for mf in self._metfilters[metfilterkey]:
            if mf in events.Flag.fields:
                metfilters = metfilters & events.Flag[mf]
        self.selections.add("metfilters", metfilters)

        self.selections.add("trigger_ele", trigger_mask["ele"])
        self.selections.add("trigger_mu", trigger_mask["mu"])
        self.selections.add("atleastone_bjet", ak.num(bjets) >= 1)
        self.selections.add("one_electron", ak.num(electrons) == 1)
        self.selections.add("one_muon", ak.num(muons) == 1)
        self.selections.add("muon_veto", ak.num(muons) == 0)
        self.selections.add("electron_veto", ak.num(electrons) == 0)
        self.selections.add("tau_veto", ak.num(taus) == 0)

        # regions
        regions = {
            "ele": {
                "numerator": [
                    "trigger_ele",
                    "trigger_mu",
                    "lumi",
                    "metfilters",
                    "atleastone_bjet",
                    "one_muon",
                    "one_electron",
                    "tau_veto"
                ],
                "denominator": [
                    "trigger_mu",
                    "lumi",
                    "metfilters",
                    "atleastone_bjet",
                    "one_muon",
                    "one_electron",
                    "tau_veto"
                ],
            },
            "mu": {
                "numerator": [
                    "trigger_ele",
                    "trigger_mu",
                    "lumi",
                    "metfilters",
                    "atleastone_bjet",
                    "one_electron",
                    "one_muon",
                    "tau_veto"
                ],
                "denominator": [
                    "trigger_ele",
                    "lumi",
                    "metfilters",
                    "atleastone_bjet",
                    "one_electron",
                    "one_muon",
                    "tau_veto"
                ],
            },
        }
        # ---------------
        # event variables
        # ---------------
        # lepton-bjet delta R and invariant mass
        ele_bjet_dr = ak.firsts(bjets).delta_r(electrons)
        ele_bjet_mass = (electrons + ak.firsts(bjets)).mass
        mu_bjet_dr = ak.firsts(bjets).delta_r(muons)
        mu_bjet_mass = (muons + ak.firsts(bjets)).mass

        # lepton-MET transverse mass
        ele_met_tranverse_mass = np.sqrt(
            2.0
            * electrons.pt
            * met.pt
            * (ak.ones_like(met.pt) - np.cos(electrons.delta_phi(met)))
        )
        mu_met_transverse_mass = np.sqrt(
            2.0
            * muons.pt
            * met.pt
            * (ak.ones_like(met.pt) - np.cos(muons.delta_phi(met)))
        )
        # lepton-bJet-MET total transverse mass
        ele_total_transverse_mass = np.sqrt(
            (electrons.pt + ak.firsts(bjets).pt + met.pt) ** 2
            - (electrons + ak.firsts(bjets) + met).pt ** 2
        )
        mu_total_transverse_mass = np.sqrt(
            (muons.pt + ak.firsts(bjets).pt + met.pt) ** 2
            - (muons + ak.firsts(bjets) + met).pt ** 2
        )
        # filling histograms
        def fill(region: str):
            selections = regions[self._lepton_flavor][region]
            region_cut = self.selections.all(*selections)
            region_weight = weights_container.weight()[region_cut]

            self.histograms["jet_kin"].fill(
                region=region,
                jet_pt=normalize(ak.firsts(bjets).pt[region_cut]),
                jet_eta=normalize(ak.firsts(bjets).eta[region_cut]),
                weight=region_weight,
            )
            self.histograms["met_kin"].fill(
                region=region,
                met_pt=normalize(met.pt[region_cut]),
                met_phi=normalize(met.phi[region_cut]),
                weight=region_weight,
            )
            self.histograms["electron_kin"].fill(
                region=region,
                electron_pt=normalize(electrons.pt[region_cut]),
                electron_eta=normalize(electrons.eta[region_cut]),
                weight=region_weight,
            )
            self.histograms["muon_kin"].fill(
                region=region,
                muon_pt=normalize(muons.pt[region_cut]),
                muon_eta=normalize(muons.eta[region_cut]),
                weight=region_weight,
            )
            self.histograms["electron_bjet_kin"].fill(
                region=region,
                electron_bjet_dr=normalize(ele_bjet_dr[region_cut]),
                invariant_mass=normalize(ele_bjet_mass[region_cut]),
                weight=region_weight,
            )
            self.histograms["muon_bjet_kin"].fill(
                region=region,
                muon_bjet_dr=normalize(mu_bjet_dr[region_cut]),
                invariant_mass=normalize(mu_bjet_mass[region_cut]),
                weight=region_weight,
            )
            self.histograms["lep_met_kin"].fill(
                region=region,
                electron_met_transverse_mass=normalize(
                    ele_met_tranverse_mass[region_cut]
                ),
                muon_met_transverse_mass=normalize(mu_met_transverse_mass[region_cut]),
                weight=region_weight,
            )
            self.histograms["lep_bjet_met_kin"].fill(
                region=region,
                electron_total_transverse_mass=normalize(
                    ele_total_transverse_mass[region_cut]
                ),
                muon_total_transverse_mass=normalize(
                    mu_total_transverse_mass[region_cut]
                ),
                weight=region_weight,
            )
            """
            # cutflow
            cutflow_selections = []
            for selection in regions[self._lepton_flavor][region]:
                cutflow_selections.append(selection)
                cutflow_cut = self.selections.all(*cutflow_selections)
                if self.is_mc:
                    cutflow_weight = weights.partial_weight(region_weights)
                    self.histograms["cutflow"][region][selection] = np.sum(
                        cutflow_weight[cutflow_cut]
                    )
                else:
                    self.histograms["cutflow"][region][selection] = np.sum(cutflow_cut)
            """

        for region in regions[self._lepton_flavor]:
            fill(region)
        output["metadata"].update({"raw_initial_nevents": nevents})
        output["histograms"] = self.histograms

        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator