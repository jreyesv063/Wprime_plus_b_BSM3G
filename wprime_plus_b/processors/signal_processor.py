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
from wprime_plus_b.corrections.jec import apply_jet_corrections, apply_fatjet_corrections
from wprime_plus_b.corrections.met import apply_met_phi_corrections, add_met_trigger_corrections
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
from wprime_plus_b.corrections.wjets_topjets import add_QCD_vs_W_weight, add_QCD_vs_Top_weight
from wprime_plus_b.corrections.tau_high_pt import add_tau_high_pt_corrections
from wprime_plus_b.corrections.ISR import ISR_weight

# Selections: Config
from wprime_plus_b.selections.signal.bjet_config import signal_bjet_selection
from wprime_plus_b.selections.signal.cases_top_tagger_config import signal_cases_selection
from wprime_plus_b.selections.signal.electron_config import signal_electron_selection
from wprime_plus_b.selections.signal.fatjet_config import signal_fatjet_selection
from wprime_plus_b.selections.signal.general_config import signal_cross_cleaning_selection, signal_trigger_selection
from wprime_plus_b.selections.signal.jet_config import signal_jet_selection
from wprime_plus_b.selections.signal.met_config import signal_met_selection
from wprime_plus_b.selections.signal.muon_config import signal_muon_selection
from wprime_plus_b.selections.signal.tau_config import signal_tau_selection
from wprime_plus_b.selections.signal.wjet_config import signal_wjet_selection

# Selections: objects
from wprime_plus_b.selections.signal.bjet_selection import select_good_bjets
from wprime_plus_b.selections.signal.electron_selection import select_good_electrons
from wprime_plus_b.selections.signal.fatjet_selection import select_good_fatjets
from wprime_plus_b.selections.signal.jet_selection import select_good_jets
from wprime_plus_b.selections.signal.muon_selection import select_good_muons
from wprime_plus_b.selections.signal.tau_selection import select_good_taus
from wprime_plus_b.selections.signal.wjet_selection import select_good_wjets

# Top tagger
from wprime_plus_b.processors.utils.topXfinder import topXfinder



from wprime_plus_b.processors.utils.analysis_utils import delta_r_mask, normalize, trigger_match, top_tagger, output_metadata, histograms_output


class SignalProccessor(processor.ProcessorABC):
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
        self.syst = syst
        self.output_type = output_type

        # define region of the analysis
        self.region = f"{self.lepton_flavor}"
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
            "top_mrec": histograms.top_tagger_hist,
            "HT": histograms.st_ht_hist,
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

                # Apply corrections only if there are jets present in at least one event
                if ak.any(ak.num(events.FatJet) > 0):
                    # fatjets
                    apply_fatjet_corrections(events, self.year)



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
                    "ele": signal_electron_selection[self.lepton_flavor]["electron_id_wp"],
                    "mu": signal_muon_selection[self.lepton_flavor]["muon_id_wp"]
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


                # ISR weights
                ISR_weight(
                    events=events, 
                    dataset=dataset, 
                    weights=weights_container, 
                    year=self.year, 
                    variation=syst_var)
                

                # add pujetid weigths
                add_pujetid_weight(
                    jets=events.Jet,
                    weights=weights_container,
                    year=self.year,
                    working_point=signal_bjet_selection[self.lepton_flavor][
                        "bjet_pileup_id"
                    ],
                    variation=syst_var,
                )
                
                # b-tagging corrector
                btag_corrector = BTagCorrector(
                    jets=events.Jet,
                    weights=weights_container,
                    sf_type="comb",
                    worging_point=signal_bjet_selection[self.lepton_flavor][
                        "btag_working_point"
                    ],
                    tagger="deepJet",
                    year=self.year,
                    full_run=False,
                    variation=syst_var,
                )
                # add b-tagging weights
                btag_corrector.add_btag_weights(flavor="b")
                btag_corrector.add_btag_weights(flavor="c")
                btag_corrector.add_btag_weights(flavor="light")
                # electron corrector
                electron_corrector = ElectronCorrector(
                    electrons=events.Electron,
                    weights=weights_container,
                    year=self.year,
                )
                # add electron ID weights
                electron_corrector.add_id_weight(
                    id_working_point=signal_electron_selection[
                        self.lepton_flavor
                    ]["electron_id_wp"]
                )
                # add electron reco weights
                electron_corrector.add_reco_weight("Above")
                electron_corrector.add_reco_weight("Below")
                # add trigger weights
                if self.lepton_flavor == "ele":
                    pass
                
                # muon corrector
                if (
                    signal_muon_selection[self.lepton_flavor]["muon_id_wp"]
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
                    id_wp=signal_muon_selection[self.lepton_flavor][
                        "muon_id_wp"
                    ],
                    iso_wp=signal_muon_selection[self.lepton_flavor][
                        "muon_iso_wp"
                    ],
                )

                # add muon RECO weights
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
                    tau_vs_jet=signal_tau_selection[self.lepton_flavor][
                        "tau_vs_jet"
                    ],
                    tau_vs_ele=signal_tau_selection[self.lepton_flavor][
                        "tau_vs_ele"
                    ],
                    tau_vs_mu=signal_tau_selection[self.lepton_flavor][
                        "tau_vs_mu"
                    ],
                    variation=syst_var,
                )
                tau_corrector.add_id_weight_DeepTau2017v2p1VSe()
                tau_corrector.add_id_weight_DeepTau2017v2p1VSmu()
                tau_corrector.add_id_weight_DeepTau2017v2p1VSjet()


                add_QCD_vs_Top_weight(
                        fatjets = events.FatJet,
                        weights = weights_container,
                        year=self.year,
                        year_mod="",
                        working_point_fatjet = signal_fatjet_selection[self.lepton_flavor]["TvsQCD"],
                        variation=syst_var
                )

                add_QCD_vs_W_weight(
                        wjets = events.FatJet,
                        weights = weights_container,
                        year=self.year,
                        year_mod="",
                        working_point_wjet = signal_wjet_selection[self.lepton_flavor]["WvsQCD"],
                        variation=syst_var
                )

                if self.lepton_flavor == "tau":

                    """
                    add_tau_high_pt_corrections(taus=events.Tau, 
                            weights=weights_container, 
                            year=self.year,
                            variation=syst_var
                    )
                    """
                    
                    with importlib.resources.path("wprime_plus_b.data", "triggers.json") as path:
                        with open(path, "r") as handle:
                            trigger_names = json.load(handle)[self.year]

                    trigger_name = trigger_names[self.lepton_flavor][0]

                    mask_trigger = (events.HLT[trigger_name])
                    # add met trigger SF
                    add_met_trigger_corrections(mask_trigger, dataset, events.MET, weights_container, self.year, "", syst_var)                    
                
                
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

            # Cross_cleaning:
            cc = signal_cross_cleaning_selection[self.lepton_flavor]["DR"]


            # select good electrons
            good_electrons = select_good_electrons(
                events=events,
                electron_pt_threshold=signal_electron_selection[
                    self.lepton_flavor
                ]["electron_pt_threshold"],
                electron_eta_threshold = signal_electron_selection[
                    self.lepton_flavor
                ]["electron_eta_threshold"],
                electron_id_wp=signal_electron_selection[
                    self.lepton_flavor
                ]["electron_id_wp"],
                electron_iso_wp=signal_electron_selection[
                    self.lepton_flavor
                ]["electron_iso_wp"],
            )
            electrons = events.Electron[good_electrons]

            # select good muons
            good_muons = select_good_muons(
                events=events,
                muon_pt_threshold=signal_muon_selection[
                    self.lepton_flavor
                ]["muon_pt_threshold"],
                muon_eta_threshold = signal_muon_selection[
                    self.lepton_flavor
                ]["muon_eta_threshold"],
                muon_id_wp= signal_muon_selection[
                    self.lepton_flavor
                ]["muon_id_wp"],
                muon_iso_wp=signal_muon_selection[
                    self.lepton_flavor
                ]["muon_iso_wp"],
            )
            good_muons = (good_muons) & (
                delta_r_mask(events.Muon, electrons, threshold=cc)
            )
            muons = events.Muon[good_muons]

            # select good taus
            good_taus = select_good_taus(
                events=events,
                tau_pt_threshold=signal_tau_selection[
                    self.lepton_flavor
                ]["tau_pt_threshold"],
                tau_eta_threshold=signal_tau_selection[
                    self.lepton_flavor
                ]["tau_eta_threshold"],
                tau_dz_threshold=signal_tau_selection[
                    self.lepton_flavor
                ]["tau_dz_threshold"],
                tau_vs_jet=signal_tau_selection[
                    self.lepton_flavor
                ]["tau_vs_jet"],
                tau_vs_ele=signal_tau_selection[
                    self.lepton_flavor
                ]["tau_vs_ele"],
                tau_vs_mu=signal_tau_selection[
                    self.lepton_flavor
                ]["tau_vs_mu"],
                prong=signal_tau_selection[
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
                btag_working_point=signal_bjet_selection[
                    self.lepton_flavor
                ]["btag_working_point"],
                jet_pt_threshold=signal_bjet_selection[
                    self.lepton_flavor
                ]["bjet_pt_threshold"],
                jet_eta_threshold = signal_bjet_selection[
                    self.lepton_flavor
                ]["bjet_eta_threshold"],
                jet_id_wp=signal_bjet_selection[
                    self.lepton_flavor
                ]["bjet_id_wp"],
                jet_pileup_id=signal_bjet_selection[
                    self.lepton_flavor
                ]["bjet_pileup_id"],
            )
            good_bjets = (
                good_bjets
                & (delta_r_mask(events.Jet, electrons, threshold=cc))
                & (delta_r_mask(events.Jet, muons, threshold=cc))
                & (delta_r_mask(events.Jet, taus, threshold=cc))
            )

            # select good jets
            good_jets = select_good_jets(
                jets=events.Jet,
                year=self.year,
                btag_working_point=signal_jet_selection[
                    self.lepton_flavor
                ]["fail_btag_working_point"],
                jet_pt_threshold=signal_jet_selection[
                    self.lepton_flavor
                ]["jet_pt_threshold"],
                jet_eta_threshold = signal_jet_selection[
                    self.lepton_flavor
                ]["jet_eta_threshold"],
                jet_id_wp=signal_jet_selection[
                    self.lepton_flavor
                ]["jet_id_wp"],
                jet_pileup_id=signal_jet_selection[
                    self.lepton_flavor
                ]["jet_pileup_id"],
            )
            good_jets = (
                good_jets
                & (delta_r_mask(events.Jet, electrons, threshold=cc))
                & (delta_r_mask(events.Jet, muons, threshold=cc))
                & (delta_r_mask(events.Jet, taus, threshold=cc))
            )

            if self.year in ["2016APV", "2016", "2018"]:
                vetomask = jetvetomaps_mask(jets=events.Jet, year=self.year, mapname="jetvetomap")
                #good_bjets = good_bjets & vetomask
                
            bjets = events.Jet[good_bjets]
            jets = events.Jet[good_jets]


            # select good fatjets: cc = 0.8
            good_fatjets = select_good_fatjets(
                fatjets = events.FatJet,
                year = self.year,
                fatjet_pt_threshold = signal_fatjet_selection[
                    self.lepton_flavor
                ]["fatjet_pt_threshold"],
                fatjet_eta_threshold = signal_fatjet_selection[
                    self.lepton_flavor
                ]["fatjet_eta_threshold"],
                TvsQCD = signal_fatjet_selection[
                    self.lepton_flavor
                ]["TvsQCD"],
            )
            good_fatjets = (
                good_fatjets
                & (delta_r_mask(events.FatJet, electrons, threshold = 2*cc))
                & (delta_r_mask(events.FatJet, muons, threshold = 2*cc))
                & (delta_r_mask(events.FatJet, taus, threshold = 2*cc))
            )   
            fatjets = events.FatJet[good_fatjets]
            

            # select good W jets
            good_wjets = select_good_wjets(
                wjets = events.FatJet,
                year = self.year,
                w_pt_threshold = signal_wjet_selection[
                    self.lepton_flavor
                ]["wjet_pt_threshold"],
                w_eta_threshold = signal_wjet_selection[
                    self.lepton_flavor
                ]["wjet_eta_threshold"],
                WvsQCD = signal_wjet_selection[
                    self.lepton_flavor
                ]["WvsQCD"],
            )
            good_wjets = (
                good_wjets
                & (delta_r_mask(events.FatJet, electrons, threshold = 2*cc))
                & (delta_r_mask(events.FatJet, muons, threshold = 2*cc))
                & (delta_r_mask(events.FatJet, taus, threshold = 2*cc))
            )   
            wjets = events.FatJet[good_wjets]
            



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
            trigger_option =  signal_trigger_selection[self.lepton_flavor]["trigger"]
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
            met_threshold =  signal_met_selection[self.lepton_flavor]["met_threshold"]
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
            self.selections.add("electron_veto", ak.num(electrons) == 0)

            self.selections.add("one_muon", ak.num(muons) == 1)
            self.selections.add("muon_veto", ak.num(muons) == 0)

            self.selections.add("one_tau", ak.num(taus) == 1)
            self.selections.add("tau_veto", ak.num(taus) == 0)

            self.selections.add("one_bjet", ak.num(bjets) == 1)
          

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


            # --------------------------
            #     Stitiching  
            # -------------------------
            # List of patterns for the datasets that should have the HT filter
            ht_filtered_datasets = [
                "DYJetsToLL_M-50_inclusive",
                "DYJetsToLL_M-50_ext",
                "WJetsToLNu_inclusive",
                "WJetsToLNu_ext"
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
               "tau": [
                    "goodvertex",
                    "lumi",
                    "Stitching",
                    f"trigger_{trigger_option}",
                    "metfilters",
                    f"met_pt_{met_threshold}",
                    "electron_veto",
                    "muon_veto",
                    "one_tau",
                    "one_bjet",
                ],
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
                    "one_muon",
                ],
            }

            # --------------
            # save cutflow before the top tagger
            # --------------
            if syst_var == "nominal":
                cut_names = region_selection[self.lepton_flavor]
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
                    *region_selection[self.lepton_flavor]
                ),
            )
            region_selection = self.selections.all(self.region)
            # check that there are events left after selection
            nevents_after = ak.sum(region_selection)

            if nevents_after == 0:
                output["metadata"]["cutflow"]["failing_top_tagger"] = ak.sum(weights_container.weight()[region_selection])
                output["metadata"]["cutflow"]["pasing_top_tagger"] = ak.sum(weights_container.weight()[region_selection])
                
                output_metadata(output = output["metadata"])
                tops = ak.zeros_like(region_selection)

            else:

                #########################
                ######### Top tagger ####
                #########################
                # Loop over the 13 cases
                cases = []
                for i in range(1, 14):
                    case_name = f"case_{i}"
                    case_value = signal_cases_selection[self.lepton_flavor].get(case_name, False)
                    if case_value:
                        cases.append(case_name)
             
                pre_bjets = bjets[region_selection]
                pre_jets = jets[region_selection]
                pre_fatjets = fatjets[region_selection]
                pre_wjets = wjets[region_selection]
                pre_electrons = electrons[region_selection]
                pre_muons = muons[region_selection]
                pre_taus = taus[region_selection]
                pre_met = events.MET[region_selection]
                pre_events = events[region_selection]


                topX = topXfinder(self.lepton_flavor, pre_bjets, pre_jets, pre_fatjets, pre_wjets, cc)
                tops, mask_top, masks = top_tagger(topX, top_tagger_cases=cases)


                #output["metadata"]["cutflow"]["failing_top_tagger"] = ak.sum(weights_container.weight()[region_selection][~mask_top])
                output["metadata"]["cutflow"]["failing_top_tagger"] = ak.sum(weights_container.weight()[region_selection][np.logical_not(mask_top)])
                output["metadata"]["cutflow"]["pasing_top_tagger"] = ak.sum(weights_container.weight()[region_selection][mask_top])


                pre_weights = weights_container.weight()[region_selection]
                
                #final_mask = ~mask_top
                final_mask = np.logical_not(mask_top)

                nevents_top_tagger = ak.sum(final_mask)
                output_metadata(output = output["metadata"], weights=pre_weights , masks=masks, mask_top= mask_top)


                if nevents_top_tagger > 0:
                    # Histograms
                    histograms_output(self, 
                                   pre_bjets, pre_jets, 
                                   pre_electrons, pre_muons, pre_taus, 
                                   pre_met, tops, 
                                   final_mask, self.lepton_flavor, self.is_mc,
                                   events)

                    if syst_var == "nominal":
                        # save weighted events to metadata
                        output["metadata"].update(
                            {
                                "weighted_final_nevents": ak.sum(
                                    pre_weights[final_mask]
                                ),
                                "raw_final_nevents": nevents_top_tagger,
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
                                    region_weight = pre_weights[final_mask]
                                else:
                                    region_weight = weights_container.weight(
                                        modifier=variation
                                    )[final_mask]
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
                            region_weight = pre_weights[final_mask]
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
                            region_weight = pre_weights[final_mask]
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
                            "weights", pre_weights[final_mask]
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