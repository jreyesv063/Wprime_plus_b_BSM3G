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
from wprime_plus_b.corrections.tau_high_pt import add_tau_high_pt_corrections

# Selections: Config
from wprime_plus_b.selections.QCD_ABCD.bjet_config import QCD_ABCD_bjet_selection
from wprime_plus_b.selections.QCD_ABCD.electron_config import QCD_ABCD_electron_selection
from wprime_plus_b.selections.QCD_ABCD.general_config import QCD_ABCD_cross_cleaning_selection, QCD_ABCD_trigger_selection
from wprime_plus_b.selections.QCD_ABCD.met_config import QCD_ABCD_met_selection
from wprime_plus_b.selections.QCD_ABCD.muon_config import QCD_ABCD_muon_selection
from wprime_plus_b.selections.QCD_ABCD.tau_config import QCD_ABCD_tau_selection
from wprime_plus_b.selections.QCD_ABCD.ditau_config import QCD_ABCD_ditau_selection
from wprime_plus_b.selections.QCD_ABCD.mt_config import QCD_ABCD_mt_selection 


# Selections: objects
from wprime_plus_b.selections.QCD_ABCD.bjet_selection import select_good_bjets
from wprime_plus_b.selections.QCD_ABCD.electron_selection import select_good_electrons
from wprime_plus_b.selections.QCD_ABCD.muon_selection import select_good_muons
from wprime_plus_b.selections.QCD_ABCD.tau_selection import select_good_taus
from wprime_plus_b.selections.QCD_ABCD.ditau_selection import select_good_ditaus
from wprime_plus_b.selections.QCD_ABCD.mt_selection import select_good_mt




from wprime_plus_b.processors.utils.analysis_utils import delta_r_mask, normalize, trigger_match



class QCD_ABCD_Proccessor(processor.ProcessorABC):

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
        self.region = f"{self.lepton_flavor}"
        # initialize dictionary of hists for control regions
        self.hist_dict = {}
        self.hist_dict[self.region] = {
            "n_kin": histograms.ttbar_n_hist,
            "met_kin": histograms.ttbar_met_hist,
            "lepton_kin": histograms.ttbar_lepton_hist,
            "lepton_bjet_kin": histograms.ttbar_lepton_bjet_hist,
            "lepton_met_kin": histograms.ttbar_lepton_met_hist,
            "lepton_met_bjet_kin": histograms.ttbar_lepton_met_bjet_hist,
            "bjet_kin": histograms.ttbar_bjet_hist,
            "tau_kin": histograms.ttbar_tau_hist,
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
                    "ele": QCD_ABCD_electron_selection[self.channel][self.lepton_flavor]["electron_id_wp"],
                    "mu": QCD_ABCD_muon_selection[self.channel][self.lepton_flavor]["muon_id_wp"]
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
                gen_weights = np.sign(events.genWeight)
                weights_container.add("genweight", gen_weights)
                
                # add l1prefiring weigths
                add_l1prefiring_weight(events, weights_container, self.year, syst_var)
                # add pileup weigths
                add_pileup_weight(events, weights_container, self.year, syst_var)
                # add pujetid weigths
                
                add_pujetid_weight(
                    jets=events.Jet,
                    weights=weights_container,
                    year=self.year,
                    working_point=QCD_ABCD_bjet_selection[self.channel][self.lepton_flavor][
                        "bjet_pileup_id"
                    ],
                    variation=syst_var,
                )
                
                # b-tagging corrector
                btag_corrector = BTagCorrector(
                    jets=events.Jet,
                    weights=weights_container,
                    sf_type="comb",
                    worging_point=QCD_ABCD_bjet_selection[self.channel][self.lepton_flavor][
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
                    id_working_point=QCD_ABCD_electron_selection[self.channel][self.lepton_flavor]["electron_id_wp"]
                )
                # add electron reco weights
                electron_corrector.add_reco_weight("Above")
                electron_corrector.add_reco_weight("Below")
                # add trigger weights
                if self.lepton_flavor == "ele":
                    pass
                
                # muon corrector
                if (
                    QCD_ABCD_muon_selection[self.channel][self.lepton_flavor]["muon_id_wp"]
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
                    id_wp=QCD_ABCD_muon_selection[self.channel][self.lepton_flavor][
                        "muon_id_wp"
                    ],
                    iso_wp=QCD_ABCD_muon_selection[self.channel][self.lepton_flavor][
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
                    tau_vs_jet=QCD_ABCD_tau_selection[self.channel][self.lepton_flavor][
                        "tau_vs_jet"
                    ],
                    tau_vs_ele=QCD_ABCD_tau_selection[self.channel][self.lepton_flavor][
                        "tau_vs_ele"
                    ],
                    tau_vs_mu=QCD_ABCD_tau_selection[self.channel][self.lepton_flavor][
                        "tau_vs_mu"
                    ],
                    variation=syst_var,
                )
                tau_corrector.add_id_weight_DeepTau2017v2p1VSe()
                tau_corrector.add_id_weight_DeepTau2017v2p1VSmu()
                tau_corrector.add_id_weight_DeepTau2017v2p1VSjet()

                if self.lepton_flavor == "tau":
                    # It is not necessary: Hight pt corrections are inside add_id_weight_DeepTau2017v2p1VSjet("pt")
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
            cc = QCD_ABCD_cross_cleaning_selection[self.channel][self.lepton_flavor]["DR"]


            # select good electrons
            good_electrons = select_good_electrons(
                events=events,
                electron_pt_threshold=QCD_ABCD_electron_selection[self.channel][
                    self.lepton_flavor
                ]["electron_pt_threshold"],
                electron_eta_threshold = QCD_ABCD_electron_selection[self.channel][
                    self.lepton_flavor
                ]["electron_eta_threshold"],
                electron_id_wp=QCD_ABCD_electron_selection[self.channel][
                    self.lepton_flavor
                ]["electron_id_wp"],
                electron_iso_wp=QCD_ABCD_electron_selection[self.channel][
                    self.lepton_flavor
                ]["electron_iso_wp"],
            )
            electrons = events.Electron[good_electrons]

            # select good muons
            good_muons = select_good_muons(
                events=events,
                muon_pt_threshold=QCD_ABCD_muon_selection[self.channel][
                    self.lepton_flavor
                ]["muon_pt_threshold"],
                muon_eta_threshold = QCD_ABCD_muon_selection[self.channel][
                    self.lepton_flavor
                ]["muon_eta_threshold"],
                muon_id_wp= QCD_ABCD_muon_selection[self.channel][
                    self.lepton_flavor
                ]["muon_id_wp"],
                muon_iso_wp=QCD_ABCD_muon_selection[self.channel][
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
                tau_pt_threshold=QCD_ABCD_tau_selection[self.channel][
                    self.lepton_flavor
                ]["tau_pt_threshold"],
                tau_eta_threshold=QCD_ABCD_tau_selection[self.channel][
                    self.lepton_flavor
                ]["tau_eta_threshold"],
                tau_dz_threshold=QCD_ABCD_tau_selection[self.channel][
                    self.lepton_flavor
                ]["tau_dz_threshold"],
                tau_vs_jet=QCD_ABCD_tau_selection[self.channel][
                    self.lepton_flavor
                ]["tau_vs_jet"],
                tau_vs_ele=QCD_ABCD_tau_selection[self.channel][
                    self.lepton_flavor
                ]["tau_vs_ele"],
                tau_vs_mu=QCD_ABCD_tau_selection[self.channel][
                    self.lepton_flavor
                ]["tau_vs_mu"],
                prong=QCD_ABCD_tau_selection[self.channel][
                    self.lepton_flavor
                ]["prongs"],
            )
            good_taus = (
                (good_taus)
                & (delta_r_mask(events.Tau, electrons, threshold=cc))
                & (delta_r_mask(events.Tau, muons, threshold=cc))
            )
            taus = events.Tau[good_taus]

            # Store wp used
            tau_wp_vs_jet = QCD_ABCD_tau_selection[self.channel][self.lepton_flavor]["tau_vs_jet"]

            # select good bjets
            good_bjets = select_good_bjets(
                jets=events.Jet,
                year=self.year,
                btag_working_point=QCD_ABCD_bjet_selection[self.channel][
                    self.lepton_flavor
                ]["btag_working_point"],
                jet_pt_threshold=QCD_ABCD_bjet_selection[self.channel][
                    self.lepton_flavor
                ]["bjet_pt_threshold"],
                jet_eta_threshold = QCD_ABCD_bjet_selection[self.channel][
                    self.lepton_flavor
                ]["bjet_eta_threshold"],
                jet_id_wp=QCD_ABCD_bjet_selection[self.channel][
                    self.lepton_flavor
                ]["bjet_id_wp"],
                jet_pileup_id=QCD_ABCD_bjet_selection[self.channel][
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


            # --------------------------
            #  mt(lepton, met) cut
            # --------------------------
            region_map = {
                "ele": electrons,
                "mu": muons,
                "tau": taus
            }
            #leptons = ak.firsts(region_map[self.lepton_flavor])
            leptons = region_map[self.lepton_flavor]

            good_mt = select_good_mt(
                    events = events,
                    lepton = leptons,
                    mt_min = QCD_ABCD_mt_selection[self.channel][self.lepton_flavor]["min_mt"],
                    mt_max = QCD_ABCD_mt_selection[self.channel][self.lepton_flavor]["max_mt"],
                    invert_mt_cut = QCD_ABCD_mt_selection[self.channel][self.lepton_flavor]["invert"],
            )


            good_ditaus = select_good_ditaus(
                    taus = taus,
                    charge_selection =  QCD_ABCD_ditau_selection[self.channel][self.lepton_flavor]["charge_tau_tau"],
                    passing_first_tau = QCD_ABCD_ditau_selection[self.channel][self.lepton_flavor]["Pass_first_tau"],
                    passing_second_tau = QCD_ABCD_ditau_selection[self.channel][self.lepton_flavor]["Pass_second_tau"],
                    failing_second_tau = QCD_ABCD_ditau_selection[self.channel][self.lepton_flavor]["Fail_second_tau"],

            )

            

            if self.year in ["2016APV", "2016", "2018"]:
                vetomask = jetvetomaps_mask(jets=events.Jet, year=self.year, mapname="jetvetomap")
                #good_bjets = good_bjets & vetomask


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
            trigger_option =  QCD_ABCD_trigger_selection[self.channel][self.lepton_flavor]["trigger"]
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
            met_threshold =  QCD_ABCD_met_selection[self.channel][self.lepton_flavor]["met_threshold"]
            self.selections.add(f"met_{met_threshold}", events.MET.pt > met_threshold)
            
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
            self.selections.add("at_least_one_muon", ak.num(muons) >= 1)

            self.selections.add(f"one_tau_{tau_wp_vs_jet}", ak.num(taus) == 1)
            self.selections.add(f"two_taus_{tau_wp_vs_jet}", ak.num(taus) == 1)
            self.selections.add("tau_veto", ak.num(taus) == 0)


            self.selections.add("bjet_veto", ak.num(bjets) == 0)

         


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
            
            # -------------------------
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

            # --------------------------
            # Fail tauvsJet wp
            # --------------------------
            with importlib.resources.open_text("wprime_plus_b.data", "tau_wps.json") as file:
                taus_wps = json.load(file)

            fail_tau_wp = QCD_ABCD_tau_selection[self.channel][self.lepton_flavor]["tau_vs_fail"]
            fail_tau_wp_mask = taus.idDeepTau2017v2p1VSjet < taus_wps["DeepTau2017"]["deep_tau_jet"][fail_tau_wp]

            self.selections.add(f"tau_fail_{fail_tau_wp}", ak.any(fail_tau_wp_mask, axis=1))
            
            # --------------------------
            #  mt(lepton, met) cut
            # --------------------------
         
            min_mt = QCD_ABCD_mt_selection[self.channel][self.lepton_flavor]["min_mt"]
            max_mt = QCD_ABCD_mt_selection[self.channel][self.lepton_flavor]["max_mt"]
            invert_mt = QCD_ABCD_mt_selection[self.channel][self.lepton_flavor]["invert"]

            self.selections.add(f"mt_cut_min_{min_mt}_and_max_{max_mt}_invert_{invert_mt}", good_mt)


            # --------------------------
            #  Di Tau
            # --------------------------
            Q_ll =  QCD_ABCD_ditau_selection[self.channel][self.lepton_flavor]["charge_tau_tau"],
            l1pass = QCD_ABCD_ditau_selection[self.channel][self.lepton_flavor]["Pass_first_tau"],
            l2pass = QCD_ABCD_ditau_selection[self.channel][self.lepton_flavor]["Pass_second_tau"],
            l2fail = QCD_ABCD_ditau_selection[self.channel][self.lepton_flavor]["Fail_second_tau"],
    
            self.selections.add(f"l1P_{l1pass}_l2P_{l2pass}_l2F_{l2fail}_charge_{Q_ll}", good_mt)

            

            # define selection regions for each channel
            region_selection = {
                "1l0b":{
                    "tau": [
                        "goodvertex",
                        "Stitching",
                        "lumi",
                        "metfilters",
                        f"trigger_{trigger_option}",
                        "HEMCleaning",
                        f"met_{met_threshold}",
                        "bjet_veto",
                        "electron_veto",
                        "muon_veto",
                        f"one_tau_{tau_wp_vs_jet}",
                        f"mt_cut_min_{min_mt}_and_max_{max_mt}_invert_{invert_mt}"
                    ],
                },
                "1l0b_A":{
                    "tau": [
                        "goodvertex",
                        "Stitching",
                        "lumi",
                        "metfilters",
                        f"trigger_{trigger_option}",
                        "HEMCleaning",
                        f"met_{met_threshold}",
                        "bjet_veto",
                        "electron_veto",
                        "muon_veto",
                        f"one_tau_{tau_wp_vs_jet}",
                        f"mt_cut_min_{min_mt}_and_max_{max_mt}_invert_{invert_mt}"
                    ],
                },
                "1l0b_B":{
                    "tau": [
                        "goodvertex",
                        "Stitching",
                        "lumi",
                        "metfilters",
                        f"trigger_{trigger_option}",
                        "HEMCleaning",
                        f"met_{met_threshold}",
                        "bjet_veto",
                        "electron_veto",
                        "muon_veto",
                        f"one_tau_{tau_wp_vs_jet}",
                        f"tau_fail_{fail_tau_wp}",
                        f"mt_cut_min_{min_mt}_and_max_{max_mt}_invert_{invert_mt}"
                    ],
                },
                "1l0b_C":{
                    "tau": [
                        "goodvertex",
                        "Stitching",
                        "lumi",
                        "metfilters",
                        f"trigger_{trigger_option}",
                        "HEMCleaning",
                        f"met_{met_threshold}",
                        "bjet_veto",
                        "electron_veto",
                        "muon_veto",
                        f"two_taus_{tau_wp_vs_jet}",
                        f"l1P_{l1pass}_l2P_{l2pass}_l2F_{l2fail}_charge_{Q_ll}",
                        f"mt_cut_min_{min_mt}_and_max_{max_mt}_invert_{invert_mt}"
                    ],
                },
                "1l0b_D":{
                    "tau": [
                        "goodvertex",
                        "Stitching",
                        "lumi",
                        "metfilters",
                        f"trigger_{trigger_option}",
                        "HEMCleaning",
                        f"met_{met_threshold}",
                        "bjet_veto",
                        "electron_veto",
                        "muon_veto",
                        f"two_taus_{tau_wp_vs_jet}",
                        f"mt_cut_min_{min_mt}_and_max_{max_mt}_invert_{invert_mt}"
                    ],
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
                region_electrons = electrons[region_selection]
                region_muons = muons[region_selection]
                region_taus = taus[region_selection]
                region_met = events.MET[region_selection]

                lepton_region_map = {
                    "ele": region_electrons,
                    "mu": region_muons,
                    "tau": region_taus
                }

                region_leptons = lepton_region_map[self.lepton_flavor]

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

                # Histograms
                self.add_feature("lepton_pt", region_leptons.pt)
                self.add_feature("lepton_eta", region_leptons.eta)
                self.add_feature("lepton_phi", region_leptons.phi)


                if self.is_mc:
                    # genPartFlav is only defined in MC samples
                    self.add_feature("genPartFlav", region_taus.genPartFlav)

                self.add_feature("decayMode", region_taus.decayMode)
                self.add_feature("isolation_electrons", region_taus.idDeepTau2017v2p1VSe)
                self.add_feature("isolation_jets", region_taus.idDeepTau2017v2p1VSjet)
                self.add_feature("isolation_muons", region_taus.idDeepTau2017v2p1VSmu)

                
                self.add_feature("bjet_pt", region_bjets.pt)
                self.add_feature("bjet_eta", region_bjets.eta)
                self.add_feature("bjet_phi", region_bjets.phi)


                self.add_feature("met",  region_met.pt)
                self.add_feature("met_phi",  region_met.phi)

                self.add_feature("lepton_bjet_dr", lepton_bjet_dr)
                self.add_feature("lepton_bjet_mass", lepton_bjet_mass)


                self.add_feature("lepton_met_mass", lepton_met_mass)
                self.add_feature("lepton_met_delta_phi", lepton_met_delta_phi)
                self.add_feature("lepton_met_bjet_mass", lepton_met_bjet_mass)

                self.add_feature("njets", ak.num(region_jets))
                self.add_feature("nbjets", ak.num(region_bjets))
                self.add_feature("npvs", events.PV.npvsGood[mask])
                self.add_feature("nmuons", ak.num(region_muons))
                self.add_feature("nelectrons", ak.num(region_electrons))
                self.add_feature("ntaus", ak.num(region_taus))



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
