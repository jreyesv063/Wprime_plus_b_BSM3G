import json
import copy
import pickle
import numpy as np
import awkward as ak
import importlib.resources
from coffea import processor
from coffea.analysis_tools import PackedSelection, Weights
from wprime_plus_b.processors.utils import histograms
from wprime_plus_b.corrections.jec import jet_corrections
from wprime_plus_b.corrections.met import met_phi_corrections
from wprime_plus_b.corrections.btag import BTagCorrector
from wprime_plus_b.corrections.pileup import add_pileup_weight
from wprime_plus_b.corrections.l1prefiring import add_l1prefiring_weight
from wprime_plus_b.corrections.pujetid import add_pujetid_weight
from wprime_plus_b.corrections.tau_energy import tau_energy_scale, met_corrected_tes
from wprime_plus_b.processors.utils.analysis_utils import delta_r_mask, normalize
from wprime_plus_b.selections.qcd.jet_selection import select_good_bjets
from wprime_plus_b.corrections.lepton import (
    ElectronCorrector,
    MuonCorrector,
    TauCorrector,
)
from wprime_plus_b.selections.qcd.config import (
    qcd_electron_selection,
    qcd_muon_selection,
    qcd_jet_selection,
    qcd_tau_selection,
)
from wprime_plus_b.selections.qcd.lepton_selection import (
    select_good_electrons,
    select_good_muons,
    select_good_taus,
)


class QcdAnalysis(processor.ProcessorABC):
    """
    QCD Analysis processor

    Parameters:
    -----------
    lepton_flavor:
        lepton flavor {'ele', 'mu'}
    year:
        year of the dataset {"2016", "2017", "2018"}
    year_mode:
        year modifier {"", "APV"}
    btag_wp:
        working point of the deepJet tagger
    syst:
        systematics to apply
    output_type:

    """

    def __init__(
        self,
        channel: str = "A",
        lepton_flavor: str = "ele",
        year: str = "2017",
        yearmod: str = "",
        output_type: str = "hist",
    ):
        self._channel = channel
        self._year = year
        self._yearmod = yearmod
        self._lepton_flavor = lepton_flavor
        self._output_type = output_type

        # initialize dictionary of hists for control regions
        self.hist_dict = {
            "met_kin": histograms.qcd_met_hist,
            "lepton_bjet_kin": histograms.qcd_lepton_bjet_hist,
            "lepton_met_kin": histograms.qcd_lepton_met_hist,
            "lepton_met_bjet_kin": histograms.qcd_lepton_met_bjet_hist,
        }
        # define dictionary to store analysis variables
        self.features = {}

    def add_feature(self, name: str, var: ak.Array) -> None:
        """add a variable array to the out dictionary"""
        self.features = {**self.features, name: var}

    def process(self, events):
        # dictionary to store output data and metadata
        output = {}
        output["metadata"] = {}

        # get dataset name
        dataset = events.metadata["dataset"]

        # get number of events before selection
        nevents = len(events)
        output["metadata"].update({"raw_initial_nevents": nevents})

        # check if sample is MC
        self.is_mc = hasattr(events, "genWeight")

        # create copies of histogram objects
        hist_dict = copy.deepcopy(self.hist_dict)

        # apply JEC/JER corrections to jets (in data, the corrections are already applied)
        if self.is_mc:
            corrected_jets, met = jet_corrections(events, self._year + self._yearmod)
        else:
            corrected_jets, met = events.Jet, events.MET
        # apply MET phi corrections
        met_pt, met_phi = met_phi_corrections(
            met_pt=met.pt,
            met_phi=met.phi,
            npvs=events.PV.npvs,
            run=events.run,
            is_mc=self.is_mc,
            year=self._year,
            year_mod=self._yearmod,
        )
        met["pt"], met["phi"] = met_pt, met_phi

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
        
        for region in ["A", "B", "C", "D"]:
            if self._channel != "all":
                if region != self._channel:
                    continue
            if (region == "A") and (dataset == "SingleMuon"):
                continue
            output["metadata"][region] = {}
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
                add_pileup_weight(
                    events, weights_container, self._year, self._yearmod, "nominal"
                )
                # add pujetid weigths
                add_pujetid_weight(
                    jets=corrected_jets,
                    weights=weights_container,
                    year=self._year,
                    year_mod=self._yearmod,
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
                    year_mod=self._yearmod,
                    full_run=False,
                    variation="nominal",
                )
                # add b-tagging weights
                btag_corrector.add_btag_weights(flavor="bc")

                # tau corrections
                tau_corrector = TauCorrector(
                    taus=corrected_taus,
                    weights=weights_container,
                    year=self._year,
                    year_mod=self._yearmod,
                    tau_vs_jet=qcd_tau_selection[region][self._lepton_flavor][
                        "tau_vs_jet"
                    ],
                    tau_vs_ele=qcd_tau_selection[region][self._lepton_flavor][
                        "tau_vs_ele"
                    ],
                    tau_vs_mu=qcd_tau_selection[region][self._lepton_flavor][
                        "tau_vs_mu"
                    ],
                    variation="nominal",
                )
                tau_corrector.add_id_weight_DeepTau2017v2p1VSe()
                tau_corrector.add_id_weight_DeepTau2017v2p1VSmu()
                tau_corrector.add_id_weight_DeepTau2017v2p1VSjet()

                """
                # electron corrector
                electron_corrector = ElectronCorrector(
                    electrons=events.Electron,
                    weights=weights_container,
                    year=self._year,
                    year_mod=self._yearmod,
                    variation="nominal",
                )
                # add electron ID weights
                electron_corrector.add_id_weight(
                    id_working_point="wp90iso"
                )
                # add electron reco weights
                electron_corrector.add_reco_weight()

                # muon corrector

                muon_corrector = MuonCorrector(
                    muons=events.Muon,
                    weights=weights_container,
                    year=self._year,
                    year_mod=self._yearmod,
                    variation="nominal",
                    id_wp="tight",
                    iso_wp="tight"
                )
                # add muon ID weights
                muon_corrector.add_id_weight()

                # add muon iso weights
                muon_corrector.add_iso_weight()

                # add trigger weights
                if self._lepton_flavor == "mu":
                    muon_corrector.add_triggeriso_weight()
                """
            # save sum of weights before selections
            output["metadata"][region]["sumw"] = ak.sum(weights_container.weight())
        
            # ------------------
            # leptons
            # -------------------
            # select good electrons
            good_electrons = select_good_electrons(
                events=events,
                region=region,
            )
            electrons = events.Electron[good_electrons]

            # select good muons
            good_muons = select_good_muons(
                events=events,
                region=region,
            )
            good_muons = (good_muons) & (
                delta_r_mask(events.Muon, electrons, threshold=0.4)
            )
            muons = events.Muon[good_muons]

            # select good taus
            good_taus = select_good_taus(
                taus=corrected_taus,
                tau_pt_threshold=qcd_tau_selection[region][self._lepton_flavor][
                    "tau_pt_threshold"
                ],
                tau_eta_threshold=qcd_tau_selection[region][self._lepton_flavor][
                    "tau_eta_threshold"
                ],
                tau_dz_threshold=qcd_tau_selection[region][self._lepton_flavor][
                    "tau_dz_threshold"
                ],
                tau_vs_jet=qcd_tau_selection[region][self._lepton_flavor][
                    "tau_vs_jet"
                ],
                tau_vs_ele=qcd_tau_selection[region][self._lepton_flavor][
                    "tau_vs_ele"
                ],
                tau_vs_mu=qcd_tau_selection[region][self._lepton_flavor][
                    "tau_vs_mu"
                ],
                prong=qcd_tau_selection[region][self._lepton_flavor]["prongs"],
            )
            good_taus = (
                (good_taus)
                & (delta_r_mask(events.Tau, electrons, threshold=0.4))
                & (delta_r_mask(events.Tau, muons, threshold=0.4))
            )
            taus = corrected_taus[good_taus]

            # ------------------
            # jets
            # -------------------
            # select good bjets
            good_bjets = select_good_bjets(
                jets=corrected_jets,
                year=self._year,
                btag_working_point=qcd_jet_selection[region][self._lepton_flavor][
                    "btag_working_point"
                ],
            )
            good_bjets = (
                good_bjets
                & (delta_r_mask(corrected_jets, electrons, threshold=0.4))
                & (delta_r_mask(corrected_jets, muons, threshold=0.4))
                & (delta_r_mask(corrected_jets, taus, threshold=0.4))
            )
            bjets = corrected_jets[good_bjets]

            # ---------------
            # event selection
            # ---------------
            # make a PackedSelection object to store selection masks
            self.selections = PackedSelection()
            
            # add luminosity calibration mask (only to data)
            with importlib.resources.path(
                "wprime_plus_b.data", "lumi_masks.pkl"
            ) as path:
                with open(path, "rb") as handle:
                    self._lumi_mask = pickle.load(handle)
            if not self.is_mc:
                lumi_mask = self._lumi_mask[self._year](
                    events.run, events.luminosityBlock
                )
            else:
                lumi_mask = np.ones(len(events), dtype="bool")
            self.selections.add("lumi", lumi_mask)

            # add lepton triggers masks
            with importlib.resources.path(
                "wprime_plus_b.data", "triggers.json"
            ) as path:
                with open(path, "r") as handle:
                    self._triggers = json.load(handle)[self._year]
            trigger = {}
            for ch in ["ele", "mu"]:
                trigger[ch] = np.zeros(nevents, dtype="bool")
                for t in self._triggers[ch]:
                    if t in events.HLT.fields:
                        trigger[ch] = trigger[ch] | events.HLT[t]
            self.selections.add("trigger_ele", trigger["ele"])
            self.selections.add("trigger_mu", trigger["mu"])

            # add MET filters mask
            with importlib.resources.path(
                "wprime_plus_b.data", "metfilters.json"
            ) as path:
                with open(path, "r") as handle:
                    self._metfilters = json.load(handle)[self._year]
            metfilters = np.ones(nevents, dtype="bool")
            metfilterkey = "mc" if self.is_mc else "data"
            for mf in self._metfilters[metfilterkey]:
                if mf in events.Flag.fields:
                    metfilters = metfilters & events.Flag[mf]
            self.selections.add("metfilters", metfilters)

            # cuts on MET
            self.selections.add("high_met_pt", met.pt > 50)
            self.selections.add("low_met_pt", met.pt < 50)

            # add number of leptons and jets
            self.selections.add("one_electron", ak.num(electrons) == 1)
            self.selections.add("electron_veto", ak.num(electrons) == 0)
            self.selections.add("one_muon", ak.num(muons) == 1)
            self.selections.add("muon_veto", ak.num(muons) == 0)
            self.selections.add("tau_veto", ak.num(taus) == 0)
            self.selections.add("one_bjet", ak.num(bjets) == 1)

            # add cut on good vertices number
            self.selections.add("goodvertex", events.PV.npvsGood > 0)

            # define selection regions for each channel
            region_selections = {
                "A": {
                    "ele": [
                        "goodvertex",
                        "lumi",
                        "trigger_ele",
                        "metfilters",
                        "high_met_pt",
                        "one_bjet",
                        "tau_veto",
                        "muon_veto",
                        "one_electron",
                    ],
                    "mu": [
                        "goodvertex",
                        "lumi",
                        "trigger_mu",
                        "metfilters",
                        "high_met_pt",
                        "one_bjet",
                        "tau_veto",
                        "electron_veto",
                        "one_muon",
                    ],
                },
                "B": {
                    "ele": [
                        "goodvertex",
                        "lumi",
                        "trigger_ele",
                        "metfilters",
                        "high_met_pt",
                        "one_bjet",
                        "tau_veto",
                        "muon_veto",
                        "one_electron",
                    ],
                    "mu": [
                        "goodvertex",
                        "lumi",
                        "trigger_mu",
                        "metfilters",
                        "high_met_pt",
                        "one_bjet",
                        "tau_veto",
                        "electron_veto",
                        "one_muon",
                    ],
                },
                "C": {
                    "ele": [
                        "goodvertex",
                        "lumi",
                        "trigger_ele",
                        "metfilters",
                        "low_met_pt",
                        "one_bjet",
                        "tau_veto",
                        "muon_veto",
                        "one_electron",
                    ],
                    "mu": [
                        "goodvertex",
                        "lumi",
                        "trigger_mu",
                        "metfilters",
                        "low_met_pt",
                        "one_bjet",
                        "tau_veto",
                        "electron_veto",
                        "one_muon",
                    ],
                },
                "D": {
                    "ele": [
                        "goodvertex",
                        "lumi",
                        "trigger_ele",
                        "metfilters",
                        "low_met_pt",
                        "one_bjet",
                        "tau_veto",
                        "muon_veto",
                        "one_electron",
                    ],
                    "mu": [
                        "goodvertex",
                        "lumi",
                        "trigger_mu",
                        "metfilters",
                        "low_met_pt",
                        "one_bjet",
                        "tau_veto",
                        "electron_veto",
                        "one_muon",
                    ],
                },
            }
            # ---------------
            # event variables
            # ---------------
            self.selections.add(
                f"{self._lepton_flavor}_{region}",
                self.selections.all(*region_selections[region][self._lepton_flavor]),
            )
            region_selection = self.selections.all(f"{self._lepton_flavor}_{region}")

            # check that there are events left after selection
            nevents_after = ak.sum(region_selection)
            if nevents_after > 0:
                # select region objects
                region_bjets = bjets[region_selection]
                region_electrons = electrons[region_selection]
                region_muons = muons[region_selection]
                region_met = met[region_selection]
                # define region leptons
                region_leptons = (
                    region_electrons if self._lepton_flavor == "ele" else region_muons
                )
                # lepton relative isolation
                lepton_reliso = (
                    region_leptons.pfRelIso04_all
                    if hasattr(region_leptons, "pfRelIso04_all")
                    else region_leptons.pfRelIso03_all
                )
                # leading bjets
                leading_bjets = ak.firsts(region_bjets)
                # lepton-bjet deltaR and invariant mass
                lepton_bjet_mass = (region_leptons + leading_bjets).mass
                # lepton-MET transverse mass and deltaPhi
                lepton_met_mass = np.sqrt(
                    2.0
                    * region_leptons.pt
                    * region_met.pt
                    * (
                        ak.ones_like(region_met.pt)
                        - np.cos(region_leptons.delta_phi(region_met))
                    )
                )
                # lepton-bJet-MET total transverse mass
                lepton_met_bjet_mass = np.sqrt(
                    (region_leptons.pt + leading_bjets.pt + region_met.pt) ** 2
                    - (region_leptons + leading_bjets + region_met).pt ** 2
                )
                self.add_feature("met", region_met.pt)
                self.add_feature("lepton_bjet_mass", lepton_bjet_mass)
                self.add_feature("lepton_met_mass", lepton_met_mass)
                self.add_feature("lepton_met_bjet_mass", lepton_met_bjet_mass)

                # ------------------
                # histogram filling
                # ------------------
                if self._output_type == "hist":
                    for kin in hist_dict:
                        # fill histograms

                        fill_args = {
                            feature: normalize(self.features[feature])
                            for feature in hist_dict[kin].axes.name[:-1]
                        }
                        hist_dict[kin].fill(
                            **fill_args,
                            region=region,
                            weight=weights_container.weight()[region_selection],
                        )
            # save metadata
            output["metadata"][region].update({"raw_final_nevents": nevents_after})
            output["metadata"][region].update(
                {
                    "weighted_final_nevents": ak.sum(
                        weights_container.weight()[region_selection]
                    )
                }
            )
        # define output dictionary accumulator
        output["histograms"] = hist_dict

        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator