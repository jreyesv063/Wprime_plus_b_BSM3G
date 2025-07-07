import json
import copy
import pickle
import numpy as np
import awkward as ak
import importlib.resources
from coffea import processor
from coffea.analysis_tools import PackedSelection, Weights
from wprime_plus_b.processors.utils import histograms


# Corrections
from wprime_plus_b.corrections.ISR import ISR_weight
from wprime_plus_b.corrections.ArbitraryWeight import ArbitraryWeight
from wprime_plus_b.corrections.ttbar_boost import add_ttbar_boost_corrections
from wprime_plus_b.corrections.jec import apply_jet_corrections, apply_fatjet_corrections
from wprime_plus_b.corrections.met import apply_met_phi_corrections, add_met_trigger_corrections
from wprime_plus_b.corrections.rochester import apply_rochester_corrections
from wprime_plus_b.corrections.tau_energy import apply_tau_energy_scale_corrections
from wprime_plus_b.corrections.pileup import add_pileup_weight
from wprime_plus_b.corrections.l1prefiring import add_l1prefiring_weight
from wprime_plus_b.corrections.pujetid import add_pujetid_weight
from wprime_plus_b.corrections.btag import BTagCorrector
from wprime_plus_b.corrections.ctag import CTagCorrector
from wprime_plus_b.corrections.muon import MuonCorrector
from wprime_plus_b.corrections.muon_highpt import MuonHighPtCorrector
from wprime_plus_b.corrections.tau import TauCorrector
from wprime_plus_b.corrections.electron import ElectronCorrector
from wprime_plus_b.corrections.jetvetomaps import jetvetomaps_mask

# Selections: Config
from wprime_plus_b.selections.ztoll.ctag_config import ztoll_cjet_config
from wprime_plus_b.selections.ztoll.bjet_config import ztoll_bjet_selection
from wprime_plus_b.selections.ztoll.electron_config import ztoll_electron_selection
from wprime_plus_b.selections.ztoll.general_config import ztoll_cross_cleaning_selection, ztoll_trigger_selection
from wprime_plus_b.selections.ztoll.leading_jet_config import ztoll_leading_jet_selection
from wprime_plus_b.selections.ztoll.jet_config import ztoll_jet_selection
from wprime_plus_b.selections.ztoll.met_config import ztoll_met_selection
from wprime_plus_b.selections.ztoll.muon_config import ztoll_muon_selection
from wprime_plus_b.selections.ztoll.tau_config import ztoll_tau_selection
from wprime_plus_b.selections.ztoll.Z_config import ztoll_charges_selection, ztoll_mrec_ll_selection


# Selections: objects
from wprime_plus_b.selections.ztoll.ctag_selection import select_good_cjets
from wprime_plus_b.selections.ztoll.bjet_selection import select_good_bjets
from wprime_plus_b.selections.ztoll.electron_selection import select_good_electrons
from wprime_plus_b.selections.ztoll.jet_selection import select_good_jets
from wprime_plus_b.selections.ztoll.leading_jet_selection import select_good_leading_jets
from wprime_plus_b.selections.ztoll.muon_selection import select_good_muons
from wprime_plus_b.selections.ztoll.tau_selection import select_good_taus
from wprime_plus_b.selections.ztoll.Z_selection import select_good_Z



from wprime_plus_b.processors.utils.analysis_utils import delta_r, delta_r_mask, normalize, trigger_match

import hist 


class ZToLLProcessor(processor.ProcessorABC):

    def __init__(
        self,
        channel: str = "ll",
        lepton_flavor: str = "ele",
        year: str = "2017",
        syst: str = "nominal",
        output_type: str = "hist",
        run_systematics: str = "false",
        output_folder: str = "",
    ):
        self.year = year
        self.lepton_flavor = lepton_flavor
        self.channel = channel
        self.syst = syst
        self.output_type = output_type

        # define region of the analysis
        self.region = f"{self.lepton_flavor}"
        # initialize dictionary of hists for control regions
        self.hist_dict = {}
        self.hist_dict[self.region] = {
            "z_boson_kin": histograms.z_boson_histogram,
            "lepton_kin": histograms.ztoll_lepton_histogram,
            "leading_lepton_kin": histograms.ztoll_leading_lepton_histogram,
            "subleading_lepton_kin": histograms.ztoll_subleading_lepton_histogram,
            "di_lepton_kin": histograms.ztoll_di_lepton_histogram,
            "cjet_kin": histograms.ztoll_cjet_histogram,
            "jet_kin": histograms.ztoll_jet_histogram,
            "met_kin": histograms.ztoll_met_histogram,
            "n_kin": histograms.ztoll_n_histogram,
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

                # Jet corrections
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

            # Consider the jet veto map correction
            if self.year in ["2016APV", "2016", "2017", "2018"]:
                vetomask = jetvetomaps_mask(jets=events.Jet, year=self.year, mapname="jetvetomap")
                # Redefine the jets with the veto applied
                events["Jet"] = events.Jet[vetomask]

            # -------------------------------------------------------------
            # event SF/weights computation
            # -------------------------------------------------------------
            # get trigger mask
            with importlib.resources.path(
                "wprime_plus_b.data", "triggers.json"
            ) as path:
                with open(path, "r") as handle:
                    self._triggers = json.load(handle)[self.year][self.lepton_flavor]

            trigger_mask = np.zeros(nevents, dtype="bool")
            # get DeltaR matched trigger objects mask
            trigger_leptons = {
                "ele": events.Electron,
                "mu": events.Muon,
            }
            trigger_match_mask = np.zeros(nevents, dtype="bool")
                     
            if self.lepton_flavor != "tau":
                lepton_id_config = {
                    "ele": ztoll_electron_selection[self.channel][self.lepton_flavor]["electron_id_wp"],
                    "mu": ztoll_muon_selection[self.channel][self.lepton_flavor]["muon_id_wp"]
                } 
                trigger_paths = self._triggers[lepton_id_config[self.lepton_flavor]]

                for tp in trigger_paths:
                    if tp in events.HLT.fields:
                        trigger_mask = trigger_mask | events.HLT[tp]

                for trigger_path in trigger_paths:
                    trig_match = trigger_match(
                        leptons=trigger_leptons[self.lepton_flavor],
                        trigobjs=events.TrigObj,
                        trigger_path=trigger_path,
                    )
                    trigger_match_mask = trigger_match_mask | trig_match
                        
            else:
                trigger_paths = self._triggers

                for tp in trigger_paths:
                    if tp in events.HLT.fields:
                        trigger_mask = trigger_mask | events.HLT[tp]
                        
                trigger_match_mask = np.ones(len(events), dtype="bool")
                    
            # -------------------------------------------------------------
            # object selection
            # -------------------------------------------------------------

            # Cross_cleaning:
            cc = ztoll_cross_cleaning_selection[self.channel][self.lepton_flavor]["DR"]


            # select good electrons
            good_electrons = select_good_electrons(
                events=events,
                electron_pt_threshold=ztoll_electron_selection[self.channel][
                    self.lepton_flavor
                ]["electron_pt_threshold"],
                electron_eta_threshold = ztoll_electron_selection[self.channel][
                    self.lepton_flavor
                ]["electron_eta_threshold"],
                electron_id_wp=ztoll_electron_selection[self.channel][
                    self.lepton_flavor
                ]["electron_id_wp"],
                electron_iso_wp=ztoll_electron_selection[self.channel][
                    self.lepton_flavor
                ]["electron_iso_wp"],
            )
            electrons = events.Electron[good_electrons]

            # select good muons
            good_muons = select_good_muons(
                events=events,
                muon_pt_threshold=ztoll_muon_selection[self.channel][
                    self.lepton_flavor
                ]["muon_pt_threshold"],
                muon_eta_threshold = ztoll_muon_selection[self.channel][
                    self.lepton_flavor
                ]["muon_eta_threshold"],
                muon_id_wp= ztoll_muon_selection[self.channel][
                    self.lepton_flavor
                ]["muon_id_wp"],
                muon_iso_wp=ztoll_muon_selection[self.channel][
                    self.lepton_flavor
                ]["muon_iso_wp"],
                year=self.year
            )
            good_muons = (good_muons) & (
                delta_r_mask(events.Muon, electrons, threshold=cc)
            )
            muons = events.Muon[good_muons]

            # select good taus
            good_taus = select_good_taus(
                events=events,
                tau_pt_threshold=ztoll_tau_selection[self.channel][
                    self.lepton_flavor
                ]["tau_pt_threshold"],
                tau_eta_threshold=ztoll_tau_selection[self.channel][
                    self.lepton_flavor
                ]["tau_eta_threshold"],
                tau_dz_threshold=ztoll_tau_selection[self.channel][
                    self.lepton_flavor
                ]["tau_dz_threshold"],
                tau_vs_jet=ztoll_tau_selection[self.channel][
                    self.lepton_flavor
                ]["tau_vs_jet"],
                tau_vs_ele=ztoll_tau_selection[self.channel][
                    self.lepton_flavor
                ]["tau_vs_ele"],
                tau_vs_mu=ztoll_tau_selection[self.channel][
                    self.lepton_flavor
                ]["tau_vs_mu"],
                prong=ztoll_tau_selection[self.channel][
                    self.lepton_flavor
                ]["prongs"],
            )
            good_taus = (
                (good_taus)
                & (delta_r_mask(events.Tau, electrons, threshold=cc))
                & (delta_r_mask(events.Tau, muons, threshold=cc))
            )
            taus = events.Tau[good_taus]

            # select good bjets
            good_bjets = select_good_bjets(
                jets=events.Jet,
                year=self.year,
                btag_working_point=ztoll_bjet_selection[self.channel][
                    self.lepton_flavor
                ]["btag_working_point"],
                jet_pt_threshold=ztoll_bjet_selection[self.channel][
                    self.lepton_flavor
                ]["bjet_pt_threshold"],
                jet_eta_threshold = ztoll_bjet_selection[self.channel][
                    self.lepton_flavor
                ]["bjet_eta_threshold"],
                jet_id_wp=ztoll_bjet_selection[self.channel][
                    self.lepton_flavor
                ]["bjet_id_wp"],
                jet_pileup_id=ztoll_bjet_selection[self.channel][
                    self.lepton_flavor
                ]["bjet_pileup_id"],
            )
            good_bjets = (
                good_bjets
                & (delta_r_mask(events.Jet, electrons, threshold=cc))
                & (delta_r_mask(events.Jet, muons, threshold=cc))
                & (delta_r_mask(events.Jet, taus, threshold=cc))
            )

            bjets = events.Jet[good_bjets]


            # select good cjets
            good_cjets = select_good_cjets(
                jets=events.Jet,
                year=self.year,
                ctag_working_point=ztoll_cjet_config[self.channel][
                    self.lepton_flavor
                ]["ctag_working_point"],
                jet_pt_threshold=ztoll_cjet_config[self.channel][
                    self.lepton_flavor
                ]["cjet_pt_threshold"],
                jet_eta_threshold =ztoll_cjet_config[self.channel][
                    self.lepton_flavor
                ]["cjet_eta_threshold"],
                jet_id_wp=ztoll_cjet_config[self.channel][
                    self.lepton_flavor
                ]["cjet_id_wp"],
                jet_pileup_id=ztoll_cjet_config[self.channel][
                    self.lepton_flavor
                ]["cjet_pileup_id"],
            )

            good_cjets = (
                good_cjets
                & (delta_r_mask(events.Jet, electrons, threshold=cc))
                & (delta_r_mask(events.Jet, muons, threshold=cc))
                & (delta_r_mask(events.Jet, taus, threshold=cc))
            )
            cjets = events.Jet[good_cjets]

            # select good jets
            good_jets = select_good_jets(
                jets=events.Jet,
                year=self.year,
                btag_working_point=ztoll_jet_selection[self.channel][
                    self.lepton_flavor
                ]["fail_btag_working_point"],
                jet_pt_threshold=ztoll_jet_selection[self.channel][
                    self.lepton_flavor
                ]["jet_pt_threshold"],
                jet_eta_threshold =ztoll_jet_selection[self.channel][
                    self.lepton_flavor
                ]["jet_eta_threshold"],
                jet_id_wp=ztoll_jet_selection[self.channel][
                    self.lepton_flavor
                ]["jet_id_wp"],
                jet_pileup_id=ztoll_jet_selection[self.channel][
                    self.lepton_flavor
                ]["jet_pileup_id"],
            )
            good_jets = (
                good_jets
                & (delta_r_mask(events.Jet, electrons, threshold=cc))
                & (delta_r_mask(events.Jet, muons, threshold=cc))
                & (delta_r_mask(events.Jet, taus, threshold=cc))
            )

            jets = events.Jet[good_jets]


            # Selec good leading Jets
            leading_jets = ak.firsts(jets)

            good_leading_jets = select_good_leading_jets(
                jets=leading_jets,
                year=self.year,
                btag_working_point=ztoll_leading_jet_selection[self.channel][
                    self.lepton_flavor
                ]["fail_btag_working_point"],
                jet_pt_threshold=ztoll_leading_jet_selection[self.channel][
                    self.lepton_flavor
                ]["jet_pt_threshold"],
                jet_eta_threshold =ztoll_leading_jet_selection[self.channel][
                    self.lepton_flavor
                ]["jet_eta_threshold"],
                jet_id_wp=ztoll_leading_jet_selection[self.channel][
                    self.lepton_flavor
                ]["jet_id_wp"],
                jet_pileup_id=ztoll_leading_jet_selection[self.channel][
                    self.lepton_flavor
                ]["jet_pileup_id"],
            )

            leading_jet =  leading_jets[good_leading_jets]


            # Selec good Z
            lepton_selection = {
                "tau": taus,
                "mu": muons,
                "ele": electrons
            }

            lepton = lepton_selection[self.lepton_flavor]

            leading_lepton = ak.pad_none(lepton, 2)[:, 0]
            subleading_lepton = ak.pad_none(lepton, 2)[:, 1]

            Z = leading_lepton  +  subleading_lepton
            good_Z = select_good_Z(
                Z,
                leading_lepton,
                subleading_lepton,
                year=self.year,
                charge_selection = ztoll_charges_selection[self.channel][self.lepton_flavor]["Charge_ll"],
                Z_mass_min = ztoll_mrec_ll_selection[self.channel][self.lepton_flavor]["m_Z_min"],
                Z_mass_max = ztoll_mrec_ll_selection[self.channel][self.lepton_flavor]["m_Z_max"]               
            )

            # set weights container
            weights_container = Weights(len(events), storeIndividual=True)

            if self.is_mc:
                # add gen weigths
                genweight_values = lambda events: np.where(events.genWeight > 0, 1, -1)
                weights_container.add("genweight", genweight_values(events))
               
                # add l1prefiring weigths
                add_l1prefiring_weight(events, weights_container, self.year, syst_var)
                # add pileup weigths
                add_pileup_weight(events, weights_container, self.year, syst_var)
                # add pujetid weigths
                
                add_pujetid_weight(
                    jets=events.Jet,
                    weights=weights_container,
                    year=self.year,
                    working_point=ztoll_bjet_selection[self.channel][self.lepton_flavor][
                        "bjet_pileup_id"
                    ],
                    variation=syst_var,
                )
                
                # b-tagging corrector
                # btag_corrector = BTagCorrector(
                #     jets=events.Jet,
                #     weights=weights_container,
                #     sf_type="comb",
                #     worging_point=ztoll_bjet_selection[self.channel][self.lepton_flavor][
                #         "btag_working_point"
                #     ],
                #     tagger="deepJet",
                #     year=self.year,
                #     full_run=False,
                #     variation=syst_var,
                # )
                # add b-tagging weights
                # btag_corrector.add_btag_weights(flavor="b")
                # btag_corrector.add_btag_weights(flavor="c")
                # btag_corrector.add_btag_weights(flavor="light")

                # c-tagging corrector
                cjet_corrector = CTagCorrector(
                    jets=events.Jet,
                    weights=weights_container,
                    working_point=ztoll_cjet_config[self.channel][self.lepton_flavor]["ctag_working_point"],
                    tagger="deepJet",
                    year=self.year,
                    variation=syst_var,
                )
                # Conditional base on the channel    
                if self.channel == "ll+c":
                    cjet_corrector.add_ctag_weights("b")
                    cjet_corrector.add_ctag_weights("c")
                    cjet_corrector.add_ctag_weights("light")


                # electron corrector
                electron_corrector = ElectronCorrector(
                    electrons=events.Electron,
                    weights=weights_container,
                    year=self.year,
                )
                # add electron ID weights
                electron_corrector.add_id_weight(
                    id_working_point=ztoll_electron_selection[self.channel][self.lepton_flavor]["electron_id_wp"]
                )
                # add electron reco weights
                electron_corrector.add_reco_weight("Above")
                electron_corrector.add_reco_weight("Below")
                # add trigger weights
                if self.lepton_flavor == "ele":
                    pass
                
                # muon corrector
                if (
                    ztoll_muon_selection[self.channel][self.lepton_flavor]["muon_id_wp"]
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
                    id_wp=ztoll_muon_selection[self.channel][self.lepton_flavor][
                        "muon_id_wp"
                    ],
                    iso_wp=ztoll_muon_selection[self.channel][self.lepton_flavor][
                        "muon_iso_wp"
                    ],
                )
                # add muon reco weights
                muon_corrector.add_reco_weight()
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
                    tau_vs_jet=ztoll_tau_selection[self.channel][self.lepton_flavor][
                        "tau_vs_jet"
                    ],
                    tau_vs_ele=ztoll_tau_selection[self.channel][self.lepton_flavor][
                        "tau_vs_ele"
                    ],
                    tau_vs_mu=ztoll_tau_selection[self.channel][self.lepton_flavor][
                        "tau_vs_mu"
                    ],
                    variation=syst_var,
                )
                tau_corrector.add_id_weight_DeepTau2017v2p1VSe()
                tau_corrector.add_id_weight_DeepTau2017v2p1VSmu()
                tau_corrector.add_id_weight_DeepTau2017v2p1VSjet()

            if syst_var == "nominal":
                # add to metadata the sum of weights without ISR weights
                output["metadata"].update({"sumw_no_ISRweight": ak.sum(weights_container.weight())})
 
                
            if self.is_mc:
                # add ISR weights
                if self.channel == "ll":
                    ISR_weight(events=events, jets=jets, dataset=dataset, weights=weights_container, year=self.year, channel=self.channel, variation=syst_var)
                
                # add Arbitrary weights
                # if self.channel == "ll+c":
                #     ArbitraryWeight(muons, jets, dataset=dataset, weights=weights_container, year=self.year, channel=self.channel, variation=syst_var)

            if syst_var == "nominal":
                # save sum of weights before selections
                output["metadata"].update({"sumw": ak.sum(weights_container.weight())})
                # save weights statistics
                output["metadata"].update({"weight_statistics": {}})
                for weight, statistics in weights_container.weightStatistics.items():
                    output["metadata"]["weight_statistics"][weight] = statistics

            # -------------------------------------------------------------
            # event selection
            # -------------------------------------------------------------
            # make a PackedSelection object to store selection masks
            self.selections = PackedSelection(dtype='uint64')
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
            trigger_option =  ztoll_trigger_selection[self.channel][self.lepton_flavor]["trigger"]
            self.selections.add(f"trigger_{trigger_option}", trigger_mask)



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
            met_threshold =  ztoll_met_selection[self.channel][self.lepton_flavor]["met_threshold"]
            self.selections.add(f"met_pt_{met_threshold}", events.MET.pt > met_threshold)
            
            # select events with at least one good vertex
            self.selections.add("goodvertex", events.PV.npvsGood > 0)

            # select events with at least one matched trigger object
            if self.lepton_flavor != "tau":
                self.selections.add(
                    "trigger_match", ak.sum(trigger_match_mask, axis=-1) > 0
                )
            else:
                self.selections.add(
                    "trigger_match", trigger_match_mask 
                )                
            # add number of leptons and jets
            self.selections.add("one_electron", ak.num(electrons) == 1)
            self.selections.add("two_electrons", ak.num(electrons) == 2)
            self.selections.add("electron_veto", ak.num(electrons) == 0)

            self.selections.add("one_muon", ak.num(muons) == 1)
            self.selections.add("two_muons", ak.num(muons) == 2)
            self.selections.add("muon_veto", ak.num(muons) == 0)

            self.selections.add("at_least_two_muons", ak.num(muons) >= 2)


            self.selections.add("one_tau", ak.num(taus) == 1)
            self.selections.add("two_taus", ak.num(taus) == 2)
            self.selections.add("tau_veto", ak.num(taus) == 0)

            self.selections.add("jet_veto", ak.num(jets) == 0)
            self.selections.add("one_jet", ak.num(jets) == 1)
            self.selections.add("two_jets", ak.num(jets) == 2)
            self.selections.add("three_jets", ak.num(jets) == 3)
            self.selections.add("more_than_three_jets", ak.num(jets) > 3)

            self.selections.add("bjet_veto", ak.num(bjets) == 0)

            self.selections.add("cjet_veto", ak.num(cjets) == 0)
            self.selections.add("one_cjet", ak.num(cjets) == 1)

            self.selections.add("at_least_one_jet", ak.num(jets) >= 1)
            self.selections.add("leading_jet", good_leading_jets) 


            self.selections.add("Z_boson", good_Z) 

            delta_R_ll = delta_r(leading_lepton, subleading_lepton, threshold = 0.4)
            self.selections.add("Delta_R_leptons", delta_R_ll)
          


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
            


            # List of patterns for the datasets that should have the HT filter
            ht_filtered_datasets = [
                "DYJetsToLL_M-50",
                "DYJetsToLL_M-50_ext1",
                "DYJetsToLL_M-10to50",
                "WJetsToLNu_inclusive",
                "WJetsToLNu_inclusive_ext1"
            ]
            # Check if the dataset starts with one of the patterns and does not contain "_HT-"
            if any(dataset.startswith(pattern) and "_HT-" not in dataset for pattern in ht_filtered_datasets):
                
                # Apply HT filter
                LowerGenHtCut = 0.0
                UpperGenHtCut = 70.0

                stitching = (
                    (events.LHE.HT >= LowerGenHtCut)
                    & (events.LHE.HT < UpperGenHtCut)
                )
                self.selections.add("Stitching", stitching)

            else: 
                self.selections.add("Stitching", np.ones(len(events), dtype="bool"))

            # define selection regions for each channel
            region_selection = {
                "ll+c": {
                    "mu": [
                        "goodvertex",
                        "lumi",
                        "metfilters",
                        f"trigger_{trigger_option}",
                        "trigger_match",
                        "HEMCleaning",
                        f"met_pt_{met_threshold}",
                        "electron_veto",
                        "tau_veto",
                        "two_muons",
                        "Delta_R_leptons",
                        "Z_boson",
                        "one_cjet",
                    ],
                },
                "ll": {
                    "mu": [
                        "goodvertex",
                        "lumi",
                        "metfilters",
                        f"trigger_{trigger_option}",
                        "trigger_match",
                        "HEMCleaning",
                        f"met_pt_{met_threshold}",
                        "electron_veto",
                        "tau_veto",
                        "two_muons",
                        "Delta_R_leptons",
                        "Z_boson",
                    ]
                },
                "ll_ISR": {
                    "mu": [
                       "goodvertex",
                       "Stitching",
                        "lumi",
                        "metfilters",
                        f"trigger_{trigger_option}",
                        "trigger_match",
                        "electron_veto",
                        "tau_veto",
                        "two_muons",
                        "bjet_veto",
                        "Z_boson",                   
                        "at_least_one_jet",
                        "leading_jet",
                    ]
                }
            }

            # --------------
            # save cutflow before the top tagger
            # --------------
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
                region_jets = jets[region_selection]
                region_electrons = electrons[region_selection]
                region_muons = muons[region_selection]
                region_taus = taus[region_selection]
                region_met = events.MET[region_selection]
                region_Z = Z[region_selection]

                #selected_objects = apply_selection(selected_objects_tmp, region_selection)

                lepton_region_map = {
                    "ele": region_electrons,
                    "mu": region_muons,
                    "tau": region_taus
                }

                region_leptons = lepton_region_map[self.lepton_flavor]

                region_leading_lepton = ak.pad_none(region_leptons, 2)[:, 0]
                region_subleading_lepton = ak.pad_none(region_leptons, 2)[:, 1]
                
                # leading bjets
                leading_bjets = ak.firsts(region_bjets)

                # Histograms
                self.add_feature("mll", region_Z.mass)
                
                self.add_feature("ptl1", region_leading_lepton.pt)
                # self.add_feature("ptl1_eta", region_leading_lepton.eta)
                # self.add_feature("ptl1_phi", region_leading_lepton.phi)

                self.add_feature("ptl2", region_subleading_lepton.pt)
                # self.add_feature("ptl2_eta", region_subleading_lepton.eta)
                # self.add_feature("ptl2_phi", region_subleading_lepton.phi)

                # self.add_feature("lepton_pt", region_leptons.pt)
                # self.add_feature("lepton_eta", region_leptons.eta)
                # self.add_feature("lepton_phi", region_leptons.phi)

                self.add_feature("ptll", (region_leading_lepton + region_subleading_lepton).pt)
                self.add_feature("dilepton_eta", (region_leading_lepton + region_subleading_lepton).eta)
                self.add_feature("dilepton_phi", (region_leading_lepton + region_subleading_lepton).phi)

                self.add_feature("njets", ak.num(region_jets)) 
                self.add_feature("npvs", events.PV.npvsGood[region_selection])
                self.add_feature("met", region_met.pt)
                self.add_feature("met_phi", region_met.phi)

                if self.channel == "ll+c":
                    self.add_feature("cjet_pt", cjets.pt)
                    self.add_feature("cjet_eta", cjets.eta)
                    self.add_feature("cjet_phi", cjets.phi)


                else:
                    self.add_feature("jet_pt", region_jets.pt)
                    self.add_feature("jet_ht", ak.sum(region_jets.pt, axis=-1))
                    self.add_feature("jet_eta", region_jets.eta)
                    self.add_feature("jet_phi", region_jets.phi)


                if syst_var == "nominal":
                    # save weighted events to metadata
                    output["metadata"].update({
                            "weighted_final_nevents": ak.sum(weights_container.weight()[region_selection]),
                            "raw_final_nevents": nevents_after,
                    })
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
                                region_weight = weights_container.weight()[region_selection]
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