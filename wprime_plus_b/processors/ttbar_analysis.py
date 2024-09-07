import json
import copy
import pickle
import numpy as np
import awkward as ak
import importlib.resources
from coffea import processor
from coffea.analysis_tools import PackedSelection, Weights
from wprime_plus_b.processors.utils import histograms
from wprime_plus_b.corrections.jec import apply_jet_corrections
from wprime_plus_b.corrections.met import apply_met_phi_corrections
from wprime_plus_b.corrections.rochester import apply_rochester_corrections
from wprime_plus_b.corrections.tau_energy import apply_tau_energy_scale_corrections
from wprime_plus_b.corrections.pileup import add_pileup_weight
from wprime_plus_b.corrections.l1prefiring import add_l1prefiring_weight
from wprime_plus_b.corrections.pujetid import add_pujetid_weight
from wprime_plus_b.corrections.btag import BTagCorrector
from wprime_plus_b.corrections.muon import MuonCorrector
from wprime_plus_b.corrections.muon_highpt import MuonHighPtCorrector
from wprime_plus_b.corrections.tau import TauCorrector
from wprime_plus_b.corrections.electron import ElectronCorrector
from wprime_plus_b.corrections.jetvetomaps import jetvetomaps_mask
from wprime_plus_b.selections.ttbar.electron_config import ttbar_electron_config
from wprime_plus_b.selections.ttbar.muon_config import ttbar_muon_config
from wprime_plus_b.selections.ttbar.tau_config import ttbar_tau_config
from wprime_plus_b.selections.ttbar.bjet_config import ttbar_bjet_config
from wprime_plus_b.selections.ttbar.electron_selection import select_good_electrons
from wprime_plus_b.selections.ttbar.muon_selection import select_good_muons
from wprime_plus_b.selections.ttbar.tau_selection import select_good_taus
from wprime_plus_b.selections.ttbar.bjet_selection import select_good_bjets
from wprime_plus_b.processors.utils.analysis_utils import delta_r_mask, normalize, trigger_match


class TtbarAnalysis(processor.ProcessorABC):
    """
    Ttbar Analysis processor

    Parameters:
    -----------
    channel:
        region channel {'2b1l', '1b1e1mu', '1b1l'}
    lepton_flavor:
        lepton flavor {'ele', 'mu'}
    year:
        year of the dataset {"2017"}
    syst:
        systematics to apply {"nominal", "jes", "jer", "met", "tau", "rochester", "full"}
    output_type:
        output object type {'hist', 'array'}
    """

    def __init__(
        self,
        channel: str = "2b1l",
        lepton_flavor: str = "ele",
        year: str = "2017",
        syst: str = "nominal",
        output_type: str = "hist",
    ):
        self.year = year
        self.lepton_flavor = lepton_flavor
        self.channel = channel
        self.syst = syst
        self.output_type = output_type

        # define region of the analysis
        self.region = f"{self.channel}_{self.lepton_flavor}"
        # initialize dictionary of hists for control regions
        self.hist_dict = {}
        self.hist_dict[self.region] = {
            "n_kin": histograms.ttbar_n_hist,
            "jet_kin": histograms.ttbar_jet_hist,
            "met_kin": histograms.ttbar_met_hist,
            "lepton_kin": histograms.ttbar_lepton_hist,
            "lepton_bjet_kin": histograms.ttbar_lepton_bjet_hist,
            "lepton_met_kin": histograms.ttbar_lepton_met_hist,
            "lepton_met_bjet_kin": histograms.ttbar_lepton_met_bjet_hist,
        }
        # define dictionary to store analysis variables
        self.features = {}
        # initialize dictionary of arrays
        self.array_dict = {}

    def add_feature(self, name: str, var: ak.Array) -> None:
        """add a variable array to the out dictionary"""
        self.features = {**self.features, name: var}

    def process(self, events):
        # get dataset name
        dataset = events.metadata["dataset"]
        # get number of events before selection
        nevents = len(events)
        # check if sample is MC
        self.is_mc = hasattr(events, "genWeight")
        # create copies of histogram objects
        hist_dict = copy.deepcopy(self.hist_dict)
        # create copy of array dictionary
        array_dict = copy.deepcopy(self.array_dict)
        # dictionary to store output data and metadata

        output = {}
        output["metadata"] = {}
        output["metadata"].update({"raw_initial_nevents": nevents})

        # define systematic variations shifts
        syst_variations = ["nominal"]
        if self.is_mc:
            jes_syst_variations = ["JESUp", "JESDown"]
            jer_syst_variations = ["JERUp", "JERDown"]
            met_syst_variations = ["UEUp", "UEDown"]
            tau_syst_variations = ["tau_up", "tau_down"]
            rochester_syst_variations = ["rochester_up", "rochester_down"]

            if self.syst == "jes":
                syst_variations.extend(jes_syst_variations)
            elif self.syst == "jer":
                syst_variations.extend(jer_syst_variations)
            elif self.syst == "met":
                syst_variations.extend(met_syst_variations)
            elif self.syst == "tau":
                syst_variations.extend(tau_syst_variations)
            elif self.syst == "rochester":
                syst_variations.extend(rochester_syst_variations)
            elif self.syst == "full":
                syst_variations.extend(jes_syst_variations)
                syst_variations.extend(jer_syst_variations)
                syst_variations.extend(met_syst_variations)
                syst_variations.extend(tau_syst_variations)
                syst_variations.extend(rochester_syst_variations)
                
        for syst_var in syst_variations:
            # -------------------------------------------------------------
            # object corrections
            # -------------------------------------------------------------
            # apply JEC/JER corrections to jets (in data, the corrections are already applied)
            if self.is_mc:
                apply_jet_corrections(events, self.year)
                # jet JEC/JER shift
                if syst_var == "JESUp":
                    events["Jet"] = events.Jet.JES_Total.up
                elif syst_var == "JESDown":
                    events["Jet"] = events.Jet.JES_Total.down
                elif syst_var == "JERUp":
                    events["Jet"] = events.Jet.JER.up
                elif syst_var == "JERDown":
                    events["Jet"] = events.Jet.JER.down
                # MET UnclusteredEnergy shift
                elif syst_var == "UEUp":
                    events["MET"] = events.MET.MET_UnclusteredEnergy.up
                elif syst_var == "UEDown":
                    events["MET"] = events.MET.MET_UnclusteredEnergy.down
                
            # apply energy corrections to taus (only to MC)
            if self.is_mc:
                apply_tau_energy_scale_corrections(
                    events=events, 
                    year=self.year, 
                    variation=syst_var
                )
            # apply rochester corretions to muons
            apply_rochester_corrections(
                events=events, 
                is_mc=self.is_mc, 
                year=self.year,
                variation=syst_var
            )
            # apply MET phi modulation corrections
            apply_met_phi_corrections(
                events=events,
                is_mc=self.is_mc,
                year=self.year,
            )
            # -------------------------------------------------------------
            # event SF/weights computation
            # -------------------------------------------------------------
            # get trigger mask
            with importlib.resources.path(
                "wprime_plus_b.data", "triggers.json"
            ) as path:
                with open(path, "r") as handle:
                    self._triggers = json.load(handle)[self.year][self.lepton_flavor]
            
            lepton_id_config = {
                "ele": ttbar_electron_config[self.channel][self.lepton_flavor]["electron_id_wp"],
                "mu": ttbar_muon_config[self.channel][self.lepton_flavor]["muon_id_wp"]
            } 
            trigger_paths = self._triggers[lepton_id_config[self.lepton_flavor]]
            trigger_mask = np.zeros(nevents, dtype="bool")
            for tp in trigger_paths:
                if tp in events.HLT.fields:
                    trigger_mask = trigger_mask | events.HLT[tp]

            # get DeltaR matched trigger objects mask
            trigger_leptons = {
                "ele": events.Electron,
                "mu": events.Muon,
            }
            trigger_match_mask = np.zeros(nevents, dtype="bool")
            for trigger_path in trigger_paths:
                trig_match = trigger_match(
                    leptons=trigger_leptons[self.lepton_flavor],
                    trigobjs=events.TrigObj,
                    trigger_path=trigger_path,
                )
                trigger_match_mask = trigger_match_mask | trig_match
                
            # set weights container
            weights_container = Weights(len(events), storeIndividual=True)
            if self.is_mc:
                # add gen weigths
                weights_container.add("genweight", events.genWeight)
                # add l1prefiring weigths
                add_l1prefiring_weight(events, weights_container, self.year, syst_var)
                # add pileup weigths
                add_pileup_weight(events, weights_container, self.year, syst_var)
                # add pujetid weigths
                add_pujetid_weight(
                    jets=events.Jet,
                    weights=weights_container,
                    year=self.year,
                    working_point=ttbar_bjet_config[self.channel][self.lepton_flavor][
                        "jet_pileup_id"
                    ],
                    variation=syst_var,
                )
                # b-tagging corrector
                btag_corrector = BTagCorrector(
                    jets=events.Jet,
                    weights=weights_container,
                    sf_type="comb",
                    worging_point=ttbar_bjet_config[self.channel][self.lepton_flavor][
                        "btag_working_point"
                    ],
                    tagger="deepJet",
                    year=self.year,
                    full_run=False,
                    variation=syst_var,
                )
                # add b-tagging weights
                btag_corrector.add_btag_weights(flavor="bc")
                btag_corrector.add_btag_weights(flavor="light")
                # electron corrector
                electron_corrector = ElectronCorrector(
                    electrons=events.Electron,
                    weights=weights_container,
                    year=self.year,
                )
                # add electron ID weights
                electron_corrector.add_id_weight(
                    id_working_point=ttbar_electron_config[self.channel][
                        self.lepton_flavor
                    ]["electron_id_wp"]
                )
                # add electron reco weights
                electron_corrector.add_reco_weight()
                # add trigger weights
                if self.lepton_flavor == "ele":
                    """
                    electron_corrector.add_trigger_weight(
                        trigger_mask=trigger_mask["ele"],
                        trigger_match_mask=trigger_match_mask
                    )
                    """
                    pass
                
                # muon corrector
                if (
                    ttbar_muon_config[self.channel][self.lepton_flavor]["muon_id_wp"]
                    == "highpt"
                ):
                    mu_corrector = MuonHighPtCorrector
                else:
                    mu_corrector = MuonCorrector
                muon_corrector = mu_corrector(
                    muons=events.Muon,
                    weights=weights_container,
                    year=self.year,
                    variation=syst_var,
                    id_wp=ttbar_muon_config[self.channel][self.lepton_flavor][
                        "muon_id_wp"
                    ],
                    iso_wp=ttbar_muon_config[self.channel][self.lepton_flavor][
                        "muon_iso_wp"
                    ],
                )
                # add muon ID weights
                muon_corrector.add_id_weight()
                # add muon iso weights
                muon_corrector.add_iso_weight()
                # add trigger weights
                if self.lepton_flavor == "mu":
                    muon_corrector.add_triggeriso_weight(
                        trigger_mask=trigger_mask,
                        trigger_match_mask=trigger_match_mask,
                    )
                
                # add tau weights
                tau_corrector = TauCorrector(
                    taus=events.Tau,
                    weights=weights_container,
                    year=self.year,
                    tau_vs_jet=ttbar_tau_config[self.channel][self.lepton_flavor][
                        "tau_vs_jet"
                    ],
                    tau_vs_ele=ttbar_tau_config[self.channel][self.lepton_flavor][
                        "tau_vs_ele"
                    ],
                    tau_vs_mu=ttbar_tau_config[self.channel][self.lepton_flavor][
                        "tau_vs_mu"
                    ],
                    variation=syst_var,
                )
                tau_corrector.add_id_weight_DeepTau2017v2p1VSe()
                tau_corrector.add_id_weight_DeepTau2017v2p1VSmu()
                tau_corrector.add_id_weight_DeepTau2017v2p1VSjet()
                
            if syst_var == "nominal":
                # save sum of weights before selections
                output["metadata"].update({"sumw": ak.sum(weights_container.weight())})
                # save weights statistics
                output["metadata"].update({"weight_statistics": {}})
                for weight, statistics in weights_container.weightStatistics.items():
                    output["metadata"]["weight_statistics"][weight] = statistics
                    
            # -------------------------------------------------------------
            # object selection
            # -------------------------------------------------------------
            # select good electrons
            good_electrons = select_good_electrons(
                events=events,
                electron_pt_threshold=ttbar_electron_config[self.channel][
                    self.lepton_flavor
                ]["electron_pt_threshold"],
                electron_id_wp=ttbar_electron_config[self.channel][
                    self.lepton_flavor
                ]["electron_id_wp"],
                electron_iso_wp=ttbar_electron_config[self.channel][
                    self.lepton_flavor
                ]["electron_iso_wp"],
            )
            electrons = events.Electron[good_electrons]

            # select good muons
            good_muons = select_good_muons(
                events=events,
                muon_pt_threshold=ttbar_muon_config[self.channel][self.lepton_flavor][
                    "muon_pt_threshold"
                ],
                muon_id_wp=ttbar_muon_config[self.channel][self.lepton_flavor][
                    "muon_id_wp"
                ],
                muon_iso_wp=ttbar_muon_config[self.channel][self.lepton_flavor][
                    "muon_iso_wp"
                ],
            )
            good_muons = (good_muons) & (
                delta_r_mask(events.Muon, electrons, threshold=0.4)
            )
            muons = events.Muon[good_muons]

            # select good taus
            good_taus = select_good_taus(
                events=events,
                tau_pt_threshold=ttbar_tau_config[self.channel][self.lepton_flavor][
                    "tau_pt_threshold"
                ],
                tau_eta_threshold=ttbar_tau_config[self.channel][self.lepton_flavor][
                    "tau_eta_threshold"
                ],
                tau_dz_threshold=ttbar_tau_config[self.channel][self.lepton_flavor][
                    "tau_dz_threshold"
                ],
                tau_vs_jet=ttbar_tau_config[self.channel][self.lepton_flavor][
                    "tau_vs_jet"
                ],
                tau_vs_ele=ttbar_tau_config[self.channel][self.lepton_flavor][
                    "tau_vs_ele"
                ],
                tau_vs_mu=ttbar_tau_config[self.channel][self.lepton_flavor][
                    "tau_vs_mu"
                ],
                prong=ttbar_tau_config[self.channel][self.lepton_flavor]["prongs"],
            )
            good_taus = (
                (good_taus)
                & (delta_r_mask(events.Tau, electrons, threshold=0.4))
                & (delta_r_mask(events.Tau, muons, threshold=0.4))
            )
            taus = events.Tau[good_taus]

            # select good bjets
            good_bjets = select_good_bjets(
                jets=events.Jet,
                year=self.year,
                btag_working_point=ttbar_bjet_config[self.channel][
                    self.lepton_flavor
                ]["btag_working_point"],
                jet_pt_threshold=ttbar_bjet_config[self.channel][self.lepton_flavor][
                    "jet_pt_threshold"
                ],
                jet_id=ttbar_bjet_config[self.channel][self.lepton_flavor]["jet_id"],
                jet_pileup_id=ttbar_bjet_config[self.channel][self.lepton_flavor][
                    "jet_pileup_id"
                ],
            )
            good_bjets = (
                good_bjets
                & (delta_r_mask(events.Jet, electrons, threshold=0.4))
                & (delta_r_mask(events.Jet, muons, threshold=0.4))
                & (delta_r_mask(events.Jet, taus, threshold=0.4))
            )
            if self.year in ["2016APV", "2016", "2018"]:
                vetomask = jetvetomaps_mask(jets=events.Jet, year=self.year, mapname="jetvetomap")
                #good_bjets = good_bjets & vetomask
                
            bjets = events.Jet[good_bjets]

            # -------------------------------------------------------------
            # event selection
            # -------------------------------------------------------------
            # make a PackedSelection object to store selection masks
            self.selections = PackedSelection()
            # add luminosity calibration mask (only to data)
            with importlib.resources.path(
                "wprime_plus_b.data", "lumi_masks.pkl"
            ) as path:
                with open(path, "rb") as handle:
                    self._lumi_mask = pickle.load(handle)
            if not self.is_mc:
                lumi_mask = self._lumi_mask[self.year](
                    events.run, events.luminosityBlock
                )
            else:
                lumi_mask = np.ones(len(events), dtype="bool")
            self.selections.add("lumi", lumi_mask)

            # add lepton triggers masks
            self.selections.add("trigger", trigger_mask)

            # add MET filters mask
            with importlib.resources.path(
                "wprime_plus_b.data", "metfilters.json"
            ) as path:
                with open(path, "r") as handle:
                    self._metfilters = json.load(handle)[self.year]
            metfilters = np.ones(nevents, dtype="bool")
            metfilterkey = "mc" if self.is_mc else "data"
            for mf in self._metfilters[metfilterkey]:
                if mf in events.Flag.fields:
                    metfilters = metfilters & events.Flag[mf]
            self.selections.add("metfilters", metfilters)

            # check that there be a minimum MET greater than 50 GeV
            self.selections.add("met_pt", events.MET.pt > 50)
            
            # select events with at least one good vertex
            self.selections.add("goodvertex", events.PV.npvsGood > 0)

            # select events with at least one matched trigger object
            self.selections.add(
                "trigger_match", ak.sum(trigger_match_mask, axis=-1) > 0
            )

            # add number of leptons and jets
            self.selections.add("one_electron", ak.num(electrons) == 1)
            self.selections.add("electron_veto", ak.num(electrons) == 0)
            self.selections.add("one_muon", ak.num(muons) == 1)
            self.selections.add("muon_veto", ak.num(muons) == 0)
            self.selections.add("tau_veto", ak.num(taus) == 0)
            self.selections.add("one_bjet", ak.num(bjets) == 1)
            self.selections.add("two_bjets", ak.num(bjets) == 2)
            
            
            if self.year == "2018":
                # hem-cleaning selection
                # https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/2000.html
                # Due to the HEM issue in year 2018, we veto the events with jets and electrons in the 
                # region -3 < eta <-1.3 and -1.57 < phi < -0.87 to remove fake MET
                hem_veto = ak.any(
                    (
                        (bjets.eta > -3.2)
                        & (bjets.eta < -1.3)
                        & (bjets.phi > -1.57)
                        & (bjets.phi < -0.87)
                    ),
                    -1,
                ) | ak.any(
                    (
                        (electrons.pt > 30)
                        & (electrons.eta > -3.2)
                        & (electrons.eta < -1.3)
                        & (electrons.phi > -1.57)
                        & (electrons.phi < -0.87)
                    ),
                    -1,
                )
                hem_cleaning = (
                    ((events.run >= 319077) & (not self.is_mc))  # if data check if in Runs C or D
                    # else for MC randomly cut based on lumi fraction of C&D
                    | ((np.random.rand(len(events)) < 0.632) & self.is_mc)
                ) & (hem_veto)

                #self.selections.add("HEMCleaning", ~hem_cleaning)
                self.selections.add("HEMCleaning", np.ones(len(events), dtype="bool"))
            else:
                self.selections.add("HEMCleaning", np.ones(len(events), dtype="bool"))
            
            # define selection regions for each channel
            region_selection = {
                "2b1l": {
                    "ele": [
                        "goodvertex",
                        "lumi",
                        "trigger",
                        "trigger_match",
                        "metfilters",
                        "HEMCleaning",
                        "met_pt",
                        "two_bjets",
                        "tau_veto",
                        "muon_veto",
                        "one_electron",
                    ],
                    "mu": [
                        "goodvertex",
                        "lumi",
                        "trigger",
                        "trigger_match",
                        "metfilters",
                        "HEMCleaning",
                        "met_pt",
                        "two_bjets",
                        "tau_veto",
                        "electron_veto",
                        "one_muon",
                    ],
                },
                "1b1e1mu": {
                    "ele": [
                        "goodvertex",
                        "lumi",
                        "trigger",
                        "trigger_match",
                        "metfilters",
                        "HEMCleaning",
                        "met_pt",
                        "one_bjet",
                        "tau_veto",
                        "one_muon",
                        "one_electron",
                    ],
                    "mu": [
                        "goodvertex",
                        "lumi",
                        "trigger",
                        "trigger_match",
                        "metfilters",
                        "HEMCleaning",
                        "met_pt",
                        "one_bjet",
                        "tau_veto",
                        "one_electron",
                        "one_muon",
                    ],
                },
                "1b1l": {
                    "ele": [
                        "goodvertex",
                        "lumi",
                        "trigger",
                        "trigger_match",
                        "metfilters",
                        "HEMCleaning",
                        "met_pt",
                        "one_bjet",
                        "tau_veto",
                        "muon_veto",
                        "one_electron",
                    ],
                    "mu": [
                        "goodvertex",
                        "lumi",
                        "trigger",
                        "trigger_match",
                        "metfilters",
                        "HEMCleaning",
                        "met_pt",
                        "one_bjet",
                        "tau_veto",
                        "electron_veto",
                        "one_muon",
                    ],
                },
            }
            # save cutflow
            if syst_var == "nominal":
                cut_names = region_selection[self.channel][self.lepton_flavor]
                output["metadata"].update({"cutflow": {}})
                selections = []
                for cut_name in cut_names:
                    selections.append(cut_name)
                    current_selection = self.selections.all(*selections)
                    output["metadata"]["cutflow"][cut_name] = ak.sum(
                        weights_container.weight()[current_selection]
                    )
            # -------------------------------------------------------------
            # event variables
            # -------------------------------------------------------------
            self.selections.add(
                self.region,
                self.selections.all(
                    *region_selection[self.channel][self.lepton_flavor]
                ),
            )
            region_selection = self.selections.all(self.region)
            # check that there are events left after selection
            nevents_after = ak.sum(region_selection)
            if nevents_after > 0:
                # select region objects
                region_bjets = bjets[region_selection]
                region_electrons = electrons[region_selection]
                region_muons = muons[region_selection]
                region_met = events.MET[region_selection]
                
                # define region leptons
                region_leptons = (
                    region_electrons if self.lepton_flavor == "ele" else region_muons
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
                lepton_bjet_dr = leading_bjets.delta_r(region_leptons)
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
                lepton_met_delta_phi = np.abs(region_leptons.delta_phi(region_met))
                # lepton-bJet-MET total transverse mass
                lepton_met_bjet_mass = np.sqrt(
                    (region_leptons.pt + leading_bjets.pt + region_met.pt) ** 2
                    - (region_leptons + leading_bjets + region_met).pt ** 2
                )
                self.add_feature("lepton_pt", region_leptons.pt)
                self.add_feature("lepton_eta", region_leptons.eta)
                self.add_feature("lepton_phi", region_leptons.phi)
                self.add_feature("jet_pt", leading_bjets.pt)
                self.add_feature("jet_eta", leading_bjets.eta)
                self.add_feature("jet_phi", leading_bjets.phi)
                self.add_feature("met", region_met.pt)
                self.add_feature("met_phi", region_met.phi)
                self.add_feature("lepton_bjet_dr", lepton_bjet_dr)
                self.add_feature("lepton_bjet_mass", lepton_bjet_mass)
                self.add_feature("lepton_met_mass", lepton_met_mass)
                self.add_feature("lepton_met_delta_phi", lepton_met_delta_phi)
                self.add_feature("lepton_met_bjet_mass", lepton_met_bjet_mass)
                self.add_feature("njets", ak.num(events.Jet)[region_selection])
                self.add_feature("npvs", events.PV.npvsGood[region_selection])

                if syst_var == "nominal":
                    # save weighted events to metadata
                    output["metadata"].update(
                        {
                            "weighted_final_nevents": ak.sum(
                                weights_container.weight()[region_selection]
                            ),
                            "raw_final_nevents": nevents_after,
                        }
                    )
                # -------------------------------------------------------------
                # histogram filling
                # -------------------------------------------------------------
                if self.output_type == "hist":
                    # break up the histogram filling for event-wise variations and object-wise variations
                    # apply event-wise variations only for nominal
                    if self.is_mc and syst_var == "nominal":
                        # get event weight systematic variations for MC samples
                        variations = ["nominal"] + list(weights_container.variations)
                        for variation in variations:
                            if variation == "nominal":
                                region_weight = weights_container.weight()[
                                    region_selection
                                ]
                            else:
                                region_weight = weights_container.weight(
                                    modifier=variation
                                )[region_selection]
                            for kin in hist_dict[self.region]:
                                fill_args = {
                                    feature: normalize(self.features[feature])
                                    for feature in hist_dict[self.region][
                                        kin
                                    ].axes.name
                                    if feature not in ["variation"]
                                }
                                hist_dict[self.region][kin].fill(
                                    **fill_args,
                                    variation=variation,
                                    weight=region_weight,
                                )
                    elif self.is_mc and syst_var != "nominal":
                        # object-wise variations
                        region_weight = weights_container.weight()[region_selection]
                        for kin in hist_dict[self.region]:
                            # get filling arguments
                            fill_args = {
                                feature: normalize(self.features[feature])
                                for feature in hist_dict[self.region][kin].axes.name[
                                    :-1
                                ]
                                if feature not in ["variation"]
                            }
                            # fill histograms
                            hist_dict[self.region][kin].fill(
                                **fill_args,
                                variation=syst_var,
                                weight=region_weight,
                            )
                    elif not self.is_mc and syst_var == "nominal":
                        # object-wise variations
                        region_weight = weights_container.weight()[region_selection]
                        for kin in hist_dict[self.region]:
                            # get filling arguments
                            fill_args = {
                                feature: normalize(self.features[feature])
                                for feature in hist_dict[self.region][kin].axes.name[
                                    :-1
                                ]
                                if feature not in ["variation"]
                            }
                            # fill histograms
                            hist_dict[self.region][kin].fill(
                                **fill_args,
                                variation=syst_var,
                                weight=region_weight,
                            )
                elif self.output_type == "array":
                    array_dict = {}
                    self.add_feature(
                        "weights", weights_container.weight()[region_selection]
                    )
                    # uncoment next two lines to save individual weights
                    # for weight in weights_container.weightStatistics:
                    #    self.add_feature(weight, weights_container.partial_weight(include=[weight]))
                    if syst_var == "nominal":
                        # select variables and put them in column accumulators
                        array_dict.update(
                            {
                                feature_name: processor.column_accumulator(
                                    normalize(feature_array)
                                )
                                for feature_name, feature_array in self.features.items()
                            }
                        )
        # define output dictionary accumulator
        if self.output_type == "hist":
            output["histograms"] = hist_dict[self.region]
        elif self.output_type == "array":
            output["arrays"] = array_dict
        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator