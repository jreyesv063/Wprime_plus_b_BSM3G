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
from wprime_plus_b.corrections.jec import apply_jet_corrections
from wprime_plus_b.corrections.met import apply_met_phi_corrections, update_met_jet_veto, met_noMu_cal, met_recoil, met_noMu_minus, met_noMu_plus
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
from wprime_plus_b.corrections.ISR import ISR_weight

# Selections: Config
from wprime_plus_b.selections.wjets.bjet_config import wjet_bjet_selection
from wprime_plus_b.selections.wjets.electron_config import wjet_electron_selection
from wprime_plus_b.selections.wjets.general_config import wjet_cross_cleaning_selection, wjet_trigger_selection
from wprime_plus_b.selections.wjets.leading_jet_config import wjet_leading_jet_selection
from wprime_plus_b.selections.wjets.jet_config import wjet_jet_selection
from wprime_plus_b.selections.wjets.met_config import wjet_met_selection
from wprime_plus_b.selections.wjets.muon_config import wjet_muon_selection
from wprime_plus_b.selections.wjets.tau_config import wjet_tau_selection
from wprime_plus_b.selections.wjets.mt_config import wjet_mt_selection 

# Selections: objects
from wprime_plus_b.selections.wjets.bjet_selection import select_good_bjets
from wprime_plus_b.selections.wjets.electron_selection import select_good_electrons
from wprime_plus_b.selections.wjets.jet_selection import select_good_jets
from wprime_plus_b.selections.wjets.leading_jet_selection import select_good_leading_jets
from wprime_plus_b.selections.wjets.muon_selection import select_good_muons
from wprime_plus_b.selections.wjets.tau_selection import select_good_taus
from wprime_plus_b.selections.wjets.delta_phi_jet_met_selection import select_good_delta_phi_jet_met
from wprime_plus_b.selections.wjets.mt_selection import select_good_mt



# Top tagger
from wprime_plus_b.processors.utils.topXfinder import topXfinder

# Systematics: Object - level
from wprime_plus_b.systematics.syst_variations import systematic_variation_mask
from wprime_plus_b.systematics.utils import update_region_map, update_region_selection


from wprime_plus_b.processors.utils.analysis_utils import delta_r_mask, normalize, trigger_match, output_metadata, histograms_output_syst_eff_wj


class WjetsProccessor(processor.ProcessorABC):

    def __init__(
        self,
        channel: str = "2b1l",
        lepton_flavor: str = "ele",
        year: str = "2017",
        syst: str = "nominal",
        output_type: str = "hist",
        run_systematics: str = "false",
        output_folder: str = ""
    ):
        self.year = year
        self.lepton_flavor = lepton_flavor
        self.channel = channel
        self.syst = syst
        self.output_type = output_type

        self.run_systematics = run_systematics

        # define region of the analysis
        self.region = f"{self.lepton_flavor}"
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
        # create copy of array dictionary
        array_dict = copy.deepcopy(self.array_dict)
        # dictionary to store output data and metadata

        output = {}
        output["metadata"] = {}
        output["metadata"].update({"raw_initial_nevents": nevents})


        # -------------------------------------------------------------
        # object corrections
        # -------------------------------------------------------------
        # apply JEC/JER corrections to jets (in data, the corrections are already applied)
        if self.is_mc:

            if self.run_systematics and self.is_mc:
                delta_list = {}

                # JER; JES and MET unclestered
                delta_list = apply_jet_corrections(events, self.year, self.run_systematics)


                # Tau energy scale corrections
                delta_list.update(apply_tau_energy_scale_corrections(events, self.year, self.run_systematics))

                # Rochester corrections
                delta_list.update(apply_rochester_corrections(events, self.is_mc, self.year, self.run_systematics))

                    
            else:
                # Jet, JER and MET unclestered corrections
                apply_jet_corrections(events, self.year, variation=self.run_systematics)


                # JET and JER for FatJets
                if ak.any(ak.num(events.FatJet) > 0):
                    # fatjets
                    apply_fatjet_corrections(events, self.year, variation=self.run_systematics)


                # Tau energy scale corrections
                apply_tau_energy_scale_corrections(
                    events=events, 
                    year=self.year, 
                    variation=self.run_systematics 
                )         

                # Rochester corrections
                apply_rochester_corrections(
                    events=events, 
                    is_mc=self.is_mc, 
                    year=self.year,
                    variation=self.run_systematics
                )

        else:    
            # Apply rochester for data
            apply_rochester_corrections(
                events=events, 
                is_mc=self.is_mc, 
                year=self.year,
                variation=self.syst
            )


        # apply MET phi modulation corrections
        apply_met_phi_corrections(
            events=events,
            is_mc=self.is_mc,
            year=self.year,
        )


        # -------------------------------------------------------------
        # Veto Jets
        # -------------------------------------------------------------
        jet_veto_mask = jetvetomaps_mask(events.Jet, self.year, "jetvetomap")
        jets_veto = events.Jet[jet_veto_mask]


        # -------------------------------------------------------------
        # object selection
        # -------------------------------------------------------------

        # Cross_cleaning:
        cc = wjet_cross_cleaning_selection[self.channel][self.lepton_flavor]["DR"]


        # select good electrons
        good_electrons = select_good_electrons(
            events=events,
            electron_pt_threshold=wjet_electron_selection[self.channel][
                self.lepton_flavor
            ]["electron_pt_threshold"],
            electron_eta_threshold = wjet_electron_selection[self.channel][
                self.lepton_flavor
            ]["electron_eta_threshold"],
            electron_id_wp=wjet_electron_selection[self.channel][
                self.lepton_flavor
            ]["electron_id_wp"],
            electron_iso_wp=wjet_electron_selection[self.channel][
                self.lepton_flavor
            ]["electron_iso_wp"],
        )
        electrons = events.Electron[good_electrons]

        # select good muons
        good_muons_masks = select_good_muons(
            events=events,
            muon_pt_threshold=wjet_muon_selection[self.channel][
                self.lepton_flavor
            ]["muon_pt_threshold"],
            muon_eta_threshold = wjet_muon_selection[self.channel][
                self.lepton_flavor
            ]["muon_eta_threshold"],
            muon_id_wp= wjet_muon_selection[self.channel][
                self.lepton_flavor
            ]["muon_id_wp"],
            muon_iso_wp=wjet_muon_selection[self.channel][
                self.lepton_flavor
            ]["muon_iso_wp"],
        )
        good_muons = (good_muons_masks["nominal"]) & (
            delta_r_mask(events.Muon, electrons, threshold=cc)
        )
        muons = events.Muon[good_muons]

        # select good taus
        good_taus_masks = select_good_taus(
            events=events,
            tau_pt_threshold=wjet_tau_selection[self.channel][
                self.lepton_flavor
            ]["tau_pt_threshold"],
            tau_eta_threshold=wjet_tau_selection[self.channel][
                self.lepton_flavor
            ]["tau_eta_threshold"],
            tau_dz_threshold=wjet_tau_selection[self.channel][
                self.lepton_flavor
            ]["tau_dz_threshold"],
            tau_vs_jet=wjet_tau_selection[self.channel][
                self.lepton_flavor
            ]["tau_vs_jet"],
            tau_vs_ele=wjet_tau_selection[self.channel][
                self.lepton_flavor
            ]["tau_vs_ele"],
            tau_vs_mu=wjet_tau_selection[self.channel][
                self.lepton_flavor
            ]["tau_vs_mu"],
            prong=wjet_tau_selection[self.channel][
                self.lepton_flavor
            ]["prongs"],
            is_mc=self.is_mc,            
        )
        good_taus = (
            (good_taus_masks["nominal"])
            & (delta_r_mask(events.Tau, electrons, threshold=cc))
            & (delta_r_mask(events.Tau, muons, threshold=cc))
        )
        taus = events.Tau[good_taus]

        # select good bjets
        good_bjets_masks = select_good_bjets(
            jets=jets_veto,
            year=self.year,
            btag_working_point=wjet_bjet_selection[self.channel][
                self.lepton_flavor
            ]["btag_working_point"],
            jet_pt_threshold=wjet_bjet_selection[self.channel][
                self.lepton_flavor
            ]["bjet_pt_threshold"],
            jet_eta_threshold = wjet_bjet_selection[self.channel][
                self.lepton_flavor
            ]["bjet_eta_threshold"],
            jet_id_wp=wjet_bjet_selection[self.channel][
                self.lepton_flavor
            ]["bjet_id_wp"],
            jet_pileup_id=wjet_bjet_selection[self.channel][
                self.lepton_flavor
            ]["bjet_pileup_id"],
            is_mc=self.is_mc,            
        )
        good_bjets = (
            good_bjets_masks["nominal"]
            & (delta_r_mask(jets_veto, electrons, threshold=cc))
            & (delta_r_mask(jets_veto, muons, threshold=cc))
            & (delta_r_mask(jets_veto, taus, threshold=cc))
        )

        bjets = jets_veto[good_bjets]

        # select good jets
        good_jets_masks = select_good_jets(
            jets=jets_veto,
            year=self.year,
            btag_working_point=wjet_jet_selection[self.channel][
                self.lepton_flavor
            ]["fail_btag_working_point"],
            jet_pt_threshold=wjet_jet_selection[self.channel][
                self.lepton_flavor
            ]["jet_pt_threshold"],
            jet_eta_threshold =wjet_jet_selection[self.channel][
                self.lepton_flavor
            ]["jet_eta_threshold"],
            jet_id_wp=wjet_jet_selection[self.channel][
                self.lepton_flavor
            ]["jet_id_wp"],
            jet_pileup_id=wjet_jet_selection[self.channel][
                self.lepton_flavor
            ]["jet_pileup_id"],
            is_mc=self.is_mc,    
            leading_jet_pt_threshold = wjet_leading_jet_selection[self.channel][
                self.lepton_flavor
            ]["jet_pt_threshold"],    
        )
        good_jets = (
            good_jets_masks["nominal"]
            & (delta_r_mask(jets_veto, electrons, threshold=cc))
            & (delta_r_mask(jets_veto, muons, threshold=cc))
            & (delta_r_mask(jets_veto, taus, threshold=cc))
            & (delta_r_mask(jets_veto, bjets, threshold=cc))
        )

        jets = jets_veto[good_jets]


        # ------------------------------------------------
        # trigger match: Only to Muons and Electrons
        # ------------------------------------------------
        # get trigger mask_match
        with importlib.resources.path(
            "wprime_plus_b.data", "triggers.json"
        ) as path:
            with open(path, "r") as handle:
                self._triggers = json.load(handle)[self.year][self.lepton_flavor]

        trigger_mask = np.zeros(nevents, dtype="bool")
        # get DeltaR matched trigger objects mask
        trigger_leptons = {
            "ele": electrons,
            "mu": muons,
        }
        trigger_match_mask = np.zeros(nevents, dtype="bool")
                    
        if self.lepton_flavor != "tau":
            lepton_id_config = {
                "ele": wjet_electron_selection[self.channel][self.lepton_flavor]["electron_id_wp"],
                "mu": wjet_muon_selection[self.channel][self.lepton_flavor]["muon_id_wp"]
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
        # Weights
        # -------------------------------------------------------------
        # set weights container
        weights_container = Weights(len(events), storeIndividual=True)

        if self.is_mc:
            # add gen weigths
            genweight_values = lambda events: np.where(events.genWeight > 0, 1, -1)
            weights_container.add("genweight", genweight_values(events))

            # add l1prefiring weigths
            add_l1prefiring_weight(events, weights_container, self.year, self.syst)
            # add pileup weigths
            add_pileup_weight(events, weights_container, self.year, self.syst)
            

            output["metadata"].update({"sumw_case_1": ak.sum(weights_container.weight())})

            # add pujetid weigths               
            add_pujetid_weight(
                jets=jets_veto,
                weights=weights_container,
                year=self.year,
                working_point=wjet_bjet_selection[self.channel][self.lepton_flavor][
                    "bjet_pileup_id"
                ],
                variation=self.syst,
            )
            
            # b-tagging corrector
            btag_corrector = BTagCorrector(
                jets=bjets,
                weights=weights_container,
                sf_type="comb",
                worging_point=wjet_bjet_selection[self.channel][self.lepton_flavor][
                    "btag_working_point"
                ],
                tagger="deepJet",
                year=self.year,
                full_run=False,
                variation=self.syst,
            )
            # add b-tagging weights
            btag_corrector.add_btag_weights(flavor="bc")
            btag_corrector.add_btag_weights(flavor="light")

            # electron corrector
            electron_corrector = ElectronCorrector(
                electrons=electrons,
                weights=weights_container,
                year=self.year,
            )
            # add electron ID weights
            electron_corrector.add_id_weight(
                id_working_point=wjet_electron_selection[self.channel][self.lepton_flavor]["electron_id_wp"]
            )
            # add electron reco weights
            electron_corrector.add_reco_weight("Above")
            electron_corrector.add_reco_weight("Below")
            # add trigger weights
            if self.lepton_flavor == "ele":
                pass
            
            # muon corrector
            if (
                wjet_muon_selection[self.channel][self.lepton_flavor]["muon_id_wp"]
                == "highpt"
            ):
                mu_corrector = MuonHighPtCorrector
            else:
                mu_corrector = MuonCorrector
                
            muon_corrector = mu_corrector(
                muons=muons,
                weights=weights_container,
                year=self.year,
                variation=self.syst,
                id_wp=wjet_muon_selection[self.channel][self.lepton_flavor][
                    "muon_id_wp"
                ],
                iso_wp=wjet_muon_selection[self.channel][self.lepton_flavor][
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
                taus=taus,
                weights=weights_container,
                year=self.year,
                tau_vs_jet=wjet_tau_selection[self.channel][self.lepton_flavor][
                    "tau_vs_jet"
                ],
                tau_vs_ele=wjet_tau_selection[self.channel][self.lepton_flavor][
                    "tau_vs_ele"
                ],
                tau_vs_mu=wjet_tau_selection[self.channel][self.lepton_flavor][
                    "tau_vs_mu"
                ],
                variation=self.syst,
            )
            tau_corrector.add_id_weight_DeepTau2017v2p1VSe()
            tau_corrector.add_id_weight_DeepTau2017v2p1VSmu()
            tau_corrector.add_id_weight_DeepTau2017v2p1VSjet()
            

            output["metadata"].update({"sumw_case_2": ak.sum(weights_container.weight())})

            # -------------------------
            # ISR correction
            # -------------------------
            ISR_weight(events=events, 
                        jets=jets_veto, 
                        dataset=dataset, 
                        weights=weights_container, 
                        year=self.year, 
                        channel=self.channel, 
                        variation=self.syst
            )

            output["metadata"].update({"sumw_case_3": ak.sum(weights_container.weight())})            

        # -------------------------
        # p_T^{miss} variables
        # -------------------------
        update_met_jet_veto(events = events, jets_veto = jets_veto)  
        met_recoil(events = events, muons = muons, electrons = electrons, taus = taus)

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
        met_threshold =  wjet_met_selection[self.channel][self.lepton_flavor]["met_threshold"]
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

        self.selections.add("one_tau", ak.num(taus) == 1)
        self.selections.add("tau_veto", ak.num(taus) == 0)


        self.selections.add("bjet_veto", ak.num(bjets) == 0)

        self.selections.add("at_least_one_jet", ak.num(jets) >= 1)          


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
                (
                    (events.run >= 319077) & (not self.is_mc)
                )  # if data check if in Runs C or D
                # else for MC randomly cut based on lumi fraction of C&D
                | ((np.random.rand(len(events)) < 0.632) & self.is_mc)
            ) & (hem_veto)

            self.selections.add("HEMCleaning", ~hem_cleaning)
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

        # -------------------------------
        # -------- Trigger: OR ----------
        # -------------------------------
        reference_trigger =  wjet_trigger_selection[self.channel][self.lepton_flavor]["trigger"]
        efficiency_trigger =  wjet_trigger_selection[self.channel][self.lepton_flavor]["trigger_eff"]
        
        if self.lepton_flavor == "mu":
            mu_id = wjet_muon_selection[self.channel][self.lepton_flavor]["muon_id_wp"]
            with importlib.resources.path("wprime_plus_b.data", "triggers.json") as path:
                with open(path, "r") as handle:
                    trigger_data = json.load(handle)
                    ref_trigger = trigger_data[self.year][reference_trigger][mu_id]
                    eff_trigger = trigger_data[self.year][efficiency_trigger]

            reference_triggers = ref_trigger


        elif self.lepton_flavor == "ele":
            ele_id = wjet_electron_selection[self.channel][self.lepton_flavor]["electron_id_wp"]
            with importlib.resources.path("wprime_plus_b.data", "triggers.json") as path:
                with open(path, "r") as handle:
                    trigger_data = json.load(handle)
                    ref_trigger = trigger_data[self.year][reference_trigger][ele_id]
                    eff_trigger = trigger_data[self.year][efficiency_trigger]
                
            reference_triggers = ref_trigger

        elif self.lepton_flavor == "tau":
            with importlib.resources.path("wprime_plus_b.data", "triggers.json") as path:
                with open(path, "r") as handle:
                    trigger_data = json.load(handle)
                    ref_trigger = trigger_data[self.year][reference_trigger]
                    eff_trigger = trigger_data[self.year][efficiency_trigger]

            reference_triggers = ref_trigger

        # Obtener los nombres reales de los triggers en el archivo HLT
        # reference_triggers = [
        #     trigger for trigger in events.HLT.fields if any(trigger.startswith(r) for r in ref_trigger)
        # ]

        efficiency_triggers = [
            trigger for trigger in events.HLT.fields if any(trigger.startswith(r) for r in eff_trigger)
        ]

        # Inicializar máscaras booleanas
        mask_reference_trigger = np.zeros(len(events), dtype="bool")
        mask_efficiency_trigger = np.zeros(len(events), dtype="bool")

        # Llenar las máscaras combinando los triggers que correspondan
        for trigger_reference in reference_triggers:
            if trigger_reference in events.HLT.fields:
                mask_reference_trigger = mask_reference_trigger | events.HLT[trigger_reference]

        for trigger_eff in efficiency_triggers:
            if trigger_eff in events.HLT.fields:
                mask_efficiency_trigger = mask_efficiency_trigger | events.HLT[trigger_eff]

        # Actualizar metadatos
        output["metadata"].update({"Triggers": reference_triggers})
        output["metadata"].update({"Triggers_eff":  efficiency_triggers})

        # Agregar selecciones
        self.selections.add(f"trigger_{reference_trigger}", mask_reference_trigger)
        self.selections.add(f"trigger_{efficiency_trigger}", mask_efficiency_trigger)


        print(f"Triggers: {reference_triggers}")



        # --------------------------
        #  mt(lepton, met) cut
        # --------------------------
        mt_cut = wjet_mt_selection[self.channel][self.lepton_flavor]["mt_threshold"]
        mt_invert = wjet_mt_selection[self.channel][self.lepton_flavor]["invert"]

        mt_mask = select_good_mt(
            met = events.MET,
            lepton = taus,
            mt_cut = mt_cut,
            invert_mt_cut = mt_invert,
        )

        self.selections.add(f"mt_{mt_cut}_invert_{mt_invert}", mt_mask)


        # --------------------------
        # deltaphi_cut cut: 
        # --------------------------    
        delta_phi_cut = wjet_met_selection[self.channel][self.lepton_flavor]["delta_phi_jet_met"]
        invert_delta_phi = wjet_met_selection[self.channel][self.lepton_flavor]["invert_delta_phi"]

        delta_phi_jet_met_mask = select_good_delta_phi_jet_met(met = events.MET, 
                                                                jets = jets, 
                                                                delta_phi_cut=delta_phi_cut, 
                                                                invert_delta_phi_cut=invert_delta_phi
                                )

        self.selections.add(f"delta_phi_jet_met_{delta_phi_cut}_{invert_delta_phi}", delta_phi_jet_met_mask)


        # define selection regions for each channel
        region_selection = {
            "1j1l": {
                "mu": [
                    "goodvertex",
                    "Stitching",
                    "lumi",
                    "metfilters",
                    "HEMCleaning",
                    f"trigger_{reference_trigger}",
                    "bjet_veto",
                    "electron_veto",
                    "tau_veto",
                    "one_muon",
                    "at_least_one_jet",
                    f"mt_{mt_cut}_invert_{mt_invert}",
                    f"delta_phi_jet_met_{delta_phi_cut}_{invert_delta_phi}",
                    f"met_{met_threshold}",
                    f"trigger_{efficiency_trigger}",
                ],
            },
            "1l0b":{
                "tau": [
                    "goodvertex",
                    "Stitching",
                    "lumi",
                    "metfilters",
                    f"trigger_{reference_trigger}",
                    "HEMCleaning",
                    "bjet_veto",
                    "electron_veto",
                    "tau_veto",
                    "one_muon",
                    "at_least_one_jet",
                    "leading_jet",
                    f"met_{met_threshold}",
                    f"delta_phi_jet_met_{delta_phi_cut}_{invert_delta_phi}",
                    f"mt_{mt_cut}_invert_{mt_invert}",
                    ],
            }
        }


        # --------------
        # save cutflow 
        # --------------
        cut_names = region_selection[self.channel][self.lepton_flavor]
        output["metadata"].update({"cutflow": {}})
        output["metadata"].update({"cutflow_raw": {}})        
        output["metadata"]["cutflow"]["sumw"] = ak.sum(weights_container.weight())
        selections = []        
        for cut_name in cut_names:
            selections.append(cut_name)
            current_selection = self.selections.all(*selections)
            output["metadata"]["cutflow"][cut_name] = ak.sum(
                weights_container.weight()[current_selection]
            )
            output["metadata"]["cutflow_raw"][cut_name] = len(
                weights_container.weight()[current_selection]
            )            
            
        # ----------------------------
        # Save weights statistics
        # ----------------------------    
        # save sum of weights before selections
        output["metadata"].update({"sumw": ak.sum(weights_container.weight())})
        # save weights statistics
        output["metadata"].update({"weight_statistics": {}})
        for weight, statistics in weights_container.weightStatistics.items():
            output["metadata"]["weight_statistics"][weight] = statistics



        # -------------------------------------------------------------
        # -------------------------------------------------------------
        #                     Systematics variations
        # -------------------------------------------------------------
        # -------------------------------------------------------------
        # If we are in MC and we want to run systematics variations
        if self.run_systematics and self.is_mc:
            # Taus, muons, bjets, light_jets, fatjets, wjets change due to object-corrections. Electrons don't have object-corrections.
            map_variation = systematic_variation_mask(events = events,
                                    lepton_flavor = self.lepton_flavor,                                                      
                                    jets_veto = jets_veto,
                                    electrons = electrons,
                                    muons = muons,
                                    taus = taus,
                                    bjets = bjets,
                                    jets = jets,
                                    muons_mask = good_muons_masks, 
                                    taus_mask = good_taus_masks, 
                                    bjets_mask = good_bjets_masks, 
                                    light_jets_mask = good_jets_masks, 
                                    delta_r_threshold = cc,
                                    met_threshold = met_threshold,
                                    mt_threshold = mt_cut,
                                    mt_inverted =  mt_invert,
                                    delta_threshold = delta_phi_cut,
                                    delta_inverted = invert_delta_phi,
                                    delta_list_met = delta_list,
                                    include_fatjets = False)




            # Names of the regions to be selected depending on the object-variation
            region_selection_cases = update_region_map(
                region_map = region_selection[self.channel][self.lepton_flavor], 
                map_variation = map_variation, 
                met_threshold = met_threshold, 
                mt_threshold =  mt_cut, 
                delta_phi = delta_phi_cut
            )


            # Store new selections in the PackedSelection object: self.selections
            selections_variations = update_region_selection(
                map_variation = map_variation, 
                selections_nominal = self.selections
            )

            # -------------------------------------------------------------
            # Save region selections: Nominal + Up/Down object-correction variations
            # -------------------------------------------------------------
            region_selection_map = {}
            self.selections.add("nominal",  self.selections.all(*region_selection[self.channel][self.lepton_flavor]))

            # Save nominal selection before the top tagger: Without systematic variations
            region_selection_map["nominal"] = self.selections.all("nominal")

            for variation, selection_names in region_selection_cases.items():
                sel_var = selections_variations[variation]
                sel_var.add(variation, sel_var.all(*selection_names))
                # Region_selection_map: Variation name as key and the selection mask as value
                region_selection_map[variation] = sel_var.all(variation)

            # --------------
            # save cutflow 
            # --------------
            for case in region_selection_cases:
                cut_names = region_selection_cases[case]
                output["metadata"].update({f"cutflow_{case}": {}})
                output["metadata"][f"cutflow_{case}"]["sumw"] = ak.sum(weights_container.weight())
                selections_var = []
                sel = self.selections if case == "nominal" else selections_variations[case]
                for cut_name in cut_names:
                    selections_var.append(cut_name)
                    current_selection = sel.all(*selections_var)
                    output["metadata"][f"cutflow_{case}"][cut_name] = ak.sum(
                        weights_container.weight()[current_selection]
                    )


            # -------------------------------------------------------------
            for region_name, region_mask in region_selection_map.items():
                # check that there are events left after selection
                nevents_after = ak.sum(region_mask)

                if region_name == "nominal":
                    output["metadata"].update({
                                f"weighted_final_nevents": ak.sum(weights_container.weight()[region_mask]),
                                f"raw_final_nevents": nevents_after,
                    })

                else:
                    output["metadata"].update({
                        f"weighted_final_nevents_{region_name}": ak.sum(weights_container.weight()[region_mask]),
                        f"raw_final_nevents_{region_name}": nevents_after,
                    })

                if nevents_after != 0:
                    # Histograms
                    histograms_output_syst_eff_wj(self, 
                                    bjets = bjets, jets = jets,
                                    electrons = electrons,  muons = muons,
                                    taus = taus, met = events.MET,
                                    mask = region_mask, 
                                    lepton_flavor = self.lepton_flavor, is_mc = self.is_mc, 
                                    events = events,
                                    syst_flag = region_name)   
                                                        
                    if self.output_type == "array":
                        # Create a dictionary to store the arrays
                        if region_name == "nominal":
                            self.add_feature(
                                f"weights", weights_container.weight()[region_mask]
                            )

                            for variation_case, weights_case in weights_container._modifiers.items():
                                self.add_feature(
                                    f"{variation_case}", weights_case[region_mask]
                                )

                            for weight in weights_container.weightStatistics:
                                filtered_weight = weights_container.partial_weight(include=[weight])[region_mask]
                                self.add_feature(weight, filtered_weight)

                        else:
                            self.add_feature(
                                f"weights_{region_name}", weights_container.weight()[region_mask]
                            )

                        # select variables and put them in column accumulators
                        array_dict.update(
                            {
                                feature_name: processor.column_accumulator(
                                    normalize(feature_array)
                                )
                                for feature_name, feature_array in self.features.items()
                            }
                        )


        else:
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
            
            region_mask = region_selection 

            output["metadata"].update({
                f"weighted_final_nevents": ak.sum(weights_container.weight()[region_mask]),
                f"raw_final_nevents": nevents_after,
            })

            if nevents_after != 0:

                tops = ak.zeros_like(region_mask)

                # Histograms
                histograms_output_syst_eff_wj(self, 
                                bjets = bjets, jets = jets,
                                electrons = electrons,  muons = muons,
                                taus = taus, met = events.MET,
                                mask = region_mask, 
                                lepton_flavor = self.lepton_flavor, is_mc = self.is_mc, 
                                events = events,
                                syst_flag = "nominal")   


                if self.output_type == "array":
                    array_dict = {}
                    self.add_feature(
                        "weights", weights_container.weight()[region_mask]
                    )


                    if self.is_mc == True:

                        if region_name == "nominal":
                            # Agregar variaciones de peso
                            for variation_case, weights_case in weights_container._modifiers.items():
                                array_dict[f"{variation_case}"] = processor.column_accumulator(
                                    weights_case[region_mask]
                                )
                            # Guardar pesos individuales filtrados por region_mask
                            for weight in weights_container.weightStatistics:
                                filtered_weight = weights_container.partial_weight(include=[weight])[region_mask]
                                self.add_feature(weight, filtered_weight)

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
        if self.output_type == "array":
            output["arrays"] = array_dict


        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator


