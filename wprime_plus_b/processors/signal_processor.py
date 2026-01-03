import re
import copy
import json
import yaml
import pickle
import numpy as np
import awkward as ak
import importlib.resources
from coffea import processor
from coffea.analysis_tools import PackedSelection, Weights

# Corrections
from wprime_plus_b.corrections.pileup import add_pileup_weight
from wprime_plus_b.corrections.pdfweights import add_pdf_weight
from wprime_plus_b.corrections.pujetid import add_pujetid_weight
from wprime_plus_b.corrections.genweight import add_genweight_weight
from wprime_plus_b.corrections.l1prefiring import add_l1prefiring_weight
from wprime_plus_b.corrections.psweights import add_particle_shower_weight
from wprime_plus_b.corrections.rochester import apply_rochester_corrections
from wprime_plus_b.corrections.top_pt_reweighting import add_TopPtReweighting
from wprime_plus_b.corrections.tau_energy import apply_tau_energy_scale_corrections
from wprime_plus_b.corrections.jec import apply_jet_corrections, apply_fatjet_corrections
from wprime_plus_b.corrections.met import apply_met_phi_corrections, add_met_trigger_corrections, update_met_jet_veto, met_recoil



from wprime_plus_b.corrections.ISR import ISR_weight
from wprime_plus_b.corrections.tau import TauCorrector
from wprime_plus_b.corrections.btag import BTagCorrector
from wprime_plus_b.corrections.muon import MuonCorrector
from wprime_plus_b.corrections.electron import ElectronCorrector
from wprime_plus_b.corrections.muon_highpt import MuonHighPtCorrector
from wprime_plus_b.corrections.top_boost import add_top_boost_corrections
from wprime_plus_b.corrections.wjets_topjets import add_QCD_vs_W_weight, add_QCD_vs_Top_weight

from wprime_plus_b.corrections.jetvetomaps import jetvetomaps_mask

# General cuts
from wprime_plus_b.general_selections.HEM import get_HEM_cleaning
from wprime_plus_b.general_selections.lumi_mask import get_lumi_mask
from wprime_plus_b.general_selections.Stitching import get_stitching_mask
from wprime_plus_b.general_selections.met_filters import get_met_filters_mask
from wprime_plus_b.general_selections.good_vertex import get_good_vertex_mask
from wprime_plus_b.general_selections.triggers import get_trigger_mask, get_trigger_match_mask


# Selections: objects
from wprime_plus_b.object_identification.tau_selection import select_good_taus
from wprime_plus_b.object_identification.muon_selection import select_good_muons
from wprime_plus_b.object_identification.bjet_selection import select_good_bjets
from wprime_plus_b.object_identification.wjet_selection import select_good_wjets
from wprime_plus_b.object_identification.fatjet_selection import select_good_fatjets
from wprime_plus_b.object_identification.lightjet_selection import select_good_lightjets
from wprime_plus_b.object_identification.electron_selection import select_good_electrons
from wprime_plus_b.object_identification.met_selection import select_good_delta_phi_jet_met, select_good_met

# Top tagger
from wprime_plus_b.processors.utils.utils_topXfinder import get_topXfinder_masks

# Systematics: Object - level
from wprime_plus_b.processors.utils.utils_syst_var import Systematic_variation

# QCD data driven
from wprime_plus_b.processors.utils.utils_qcd_data_driven import QCD_data_driven

# Utils
from wprime_plus_b.processors.utils.histogram_utils import histograms_output_array
from wprime_plus_b.processors.utils.analysis_utils import delta_r_mask, normalize, output_metadata, histograms_output_syst, fill_cutflow


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
        processor: str = "top_tagger",
        channel: str = "2b1l",
        lepton_flavor: str = "ele",
        year: str = "2017",
        syst: str = "nominal",
        output_type: str = "hist",
        run_systematics: str = "false",
        qcd_data_driven: str = "false",
        output_folder: str = ""
    ):
        self.run_systematics = run_systematics 
        self.year = year
        self.lepton_flavor = lepton_flavor
        self.syst = syst
        self.output_type = output_type
        self.processor = processor

        # define dictionary to store analysis variables
        self.features = {}
        # initialize dictionary of arrays
        self.array_dict = {}

        # Load event selection criteria
        with open(f"wprime_plus_b/selections/{processor}/event_selection_criteria.yaml") as f:
            self.criteria = yaml.safe_load(f)

        self.qcd_data_driven = (
            qcd_data_driven == "true" and
            all(field in self.criteria.get("data_driven_qcd_estimation", {}) for field in ["cr_b", "cr_c", "cr_d"])
        )


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

                # JER and JES for FatJets.
                if ak.any(ak.num(events.FatJet) > 0):
                    delta_list.update(apply_fatjet_corrections(events, self.year, self.run_systematics))

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
                
        # ===============================================================
        #            Object identification
        # ==============================================================
        # Cross cleaning
        cc = self.criteria["cross_cleaning"][self.lepton_flavor]

        # Select good electrons
        good_electrons = select_good_electrons(
            events=events,
            electron_pt_threshold= self.criteria["electron"][self.lepton_flavor]["pt"],
            electron_eta_threshold = self.criteria["electron"][self.lepton_flavor]["eta"],
            electron_id_wp= self.criteria["electron"][self.lepton_flavor]["id"],
            electron_iso_wp=self.criteria["electron"][self.lepton_flavor]["iso"],
        )
        electrons = events.Electron[good_electrons]


        # select good muons
        good_muons_masks = select_good_muons(
            events=events,
            muon_pt_threshold = self.criteria["muon"][self.lepton_flavor]["pt"],
            muon_eta_threshold = self.criteria["muon"][self.lepton_flavor]["eta"],
            muon_id_wp = self.criteria["muon"][self.lepton_flavor]["id"],
            muon_iso_wp = self.criteria["muon"][self.lepton_flavor]["iso"],
        )
        good_muons = (good_muons_masks["nominal"]) & (
            delta_r_mask(events.Muon, electrons, threshold=cc)
        )
        muons = events.Muon[good_muons]
        

        # select good taus
        good_taus_masks = select_good_taus(
            events=events,
            tau_pt_threshold = self.criteria["tau"][self.lepton_flavor]["pt"],
            tau_eta_threshold = self.criteria["tau"][self.lepton_flavor]["eta"],
            tau_dz_threshold = self.criteria["tau"][self.lepton_flavor]["dz"],
            tau_vs_jet_pass = self.criteria["tau"][self.lepton_flavor]["fake_VSjet_pass"],
            tau_vs_ele =self.criteria["tau"][self.lepton_flavor]["fake_VSe"],
            tau_vs_mu = self.criteria["tau"][self.lepton_flavor]["fake_VSmu"],
            prong = self.criteria["tau"][self.lepton_flavor]["prongs"] ,
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
            events=events,
            jets=jets_veto ,
            year=self.year,
            btag_working_point_pass = self.criteria["bjet"][self.lepton_flavor]["btag_wp_pass"],
            jet_pt_threshold = self.criteria["bjet"][self.lepton_flavor]["pt"],
            jet_eta_threshold = self.criteria["bjet"][self.lepton_flavor]["eta"],
            jet_id_wp = self.criteria["bjet"][self.lepton_flavor]["jet_id"],
            jet_pileup_id = self.criteria["bjet"][self.lepton_flavor]["pileup_id"],
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
        good_jets_masks = select_good_lightjets(
            events= events,
            jets=jets_veto,
            year=self.year,
            btag_working_point_fail = self.criteria["jet"][self.lepton_flavor]["btag_wp_fail"],
            jet_pt_threshold = self.criteria["jet"][self.lepton_flavor]["pt"],
            jet_eta_threshold = self.criteria["jet"][self.lepton_flavor]["eta"],
            jet_id_wp = self.criteria["jet"][self.lepton_flavor]["jet_id"],
            jet_pileup_id = self.criteria["jet"][self.lepton_flavor]["pileup_id"],
            is_mc=self.is_mc,
        )
        good_jets = (
            good_jets_masks["nominal"]
            & (delta_r_mask(jets_veto, electrons, threshold=cc))
            & (delta_r_mask(jets_veto, muons, threshold=cc))
            & (delta_r_mask(jets_veto, taus, threshold=cc))
        )
        jets = jets_veto[good_jets]

        # select good fatjets: cc = 0.8
        good_fatjets_masks = select_good_fatjets(
            fatjets = events.FatJet,
            year = self.year,
            fatjet_pt_threshold = self.criteria["fatjet"][self.lepton_flavor]["pt"],
            fatjet_eta_threshold = self.criteria["fatjet"][self.lepton_flavor]["eta"],
            TvsQCD = self.criteria["fatjet"][self.lepton_flavor]["particleNet_Top_Nominal"],
            is_mc=self.is_mc,
        )
        good_fatjets = (
            good_fatjets_masks["nominal"]
            & (delta_r_mask(events.FatJet, electrons, threshold = 2*cc))
            & (delta_r_mask(events.FatJet, muons, threshold = 2*cc))
            & (delta_r_mask(events.FatJet, taus, threshold = 2*cc))
            & (delta_r_mask(events.FatJet, bjets, threshold = 2*cc))
            & (delta_r_mask(events.FatJet, jets, threshold = 2*cc))
        )   
        fatjets = events.FatJet[good_fatjets]

        
        # select good W jets
        good_wjets_masks = select_good_wjets(
            wjets = events.FatJet,
            year = self.year,
            w_pt_threshold = self.criteria["wjet"][self.lepton_flavor]["pt"],
            w_eta_threshold = self.criteria["wjet"][self.lepton_flavor]["eta"],
            WvsQCD = self.criteria["wjet"][self.lepton_flavor]["particleNet_W_Nominal"],
            is_mc=self.is_mc,
        )
        good_wjets = (
            good_wjets_masks["nominal"]
            & (delta_r_mask(events.FatJet, electrons, threshold = 2*cc))
            & (delta_r_mask(events.FatJet, muons, threshold = 2*cc))
            & (delta_r_mask(events.FatJet, taus, threshold = 2*cc))
            & (delta_r_mask(events.FatJet, bjets, threshold = 2*cc))
            & (delta_r_mask(events.FatJet, jets, threshold = 2*cc))
            & (delta_r_mask(events.FatJet, fatjets, threshold = 2*cc))
        )   
        wjets = events.FatJet[good_wjets]

        # --------------------------------
        # Missing transverse momentum (MET)
        # --------------------------------
        # Update MET by applying jet veto corrections
        update_met_jet_veto(events = events, jets_veto = jets_veto)  
        # Compute recoil-corrected MET using reconstructed leptons
        met_recoil(events = events, muons = muons)

        
        # Select good MET
        good_met_masks = select_good_met(
            events = events,
            met_min = self.criteria["met"][self.lepton_flavor]["met_min"],
            year = self.year,
        )


        # Create a dictionary of objects to simplify handling
        objects = {
            "bjets": bjets,
            "jets": jets,
            "jets_veto": jets_veto,              
            "fatjets": fatjets,
            "wjets": wjets,
            "electrons": electrons,
            "muons": muons,
            "taus": taus,
            "met": events.MET,
            "events": events
        }


        objects_variations = {
            "bjets": good_bjets_masks,
            "jets": good_jets_masks,
            "fatjets": good_fatjets_masks,
            "wjets": good_wjets_masks,
            "electrons": None,
            "muons": good_muons_masks,
            "taus": good_taus_masks,
        }
        
        # ==============================================================
        #           Event-level weights
        # ==============================================================
        # set weights container
        weights_container = Weights(len(events), storeIndividual=True)
        
        if self.is_mc:
            # add gen weigths
            add_genweight_weight(events, weights_container)

            # add l1prefiring weigths
            add_l1prefiring_weight(events, weights_container, self.year, self.syst)

            # add ps weigths
            add_particle_shower_weight(events, weights_container, self.year, self.syst)

            # add pdf weigths
            add_pdf_weight(events, weights_container, self.year, variation=self.syst, dataset = dataset)

            # add top pt reweighting
            add_TopPtReweighting(events, weights_container, dataset, self.syst)


            # add pileup weigths
            add_pileup_weight(events, weights_container, self.year, self.syst)

            output["metadata"].update({"sumw_no_object_weights": ak.sum(weights_container.weight())})

            # add pujetid weigths
            add_pujetid_weight(
                jets=jets_veto,
                weights=weights_container,
                year=self.year,
                working_point = self.criteria["jet"][self.lepton_flavor]["pileup_id"],
                variation=self.syst,
            )
            
            # b-tagging corrector
            btag_corrector = BTagCorrector(
                jets=objects["bjets"],
                weights=weights_container,
                sf_type="comb",
                working_point = self.criteria["bjet"][self.lepton_flavor]["btag_wp_pass"],
                tagger="deepJet",
                year=self.year,
                full_run=False,
                variation=self.syst,
                dataset = dataset
            )

            # add b-tagging weights
            btag_corrector.add_btag_weights(flavor="bc")
            btag_corrector.add_btag_weights(flavor="light")
            #btag_corrector.print_efficiency_min_max_per_flavor()

            # electron corrector
            electron_corrector = ElectronCorrector(
                electrons=objects["electrons"],
                weights=weights_container,
                year=self.year,
            )

            # add electron ID weights
            electron_corrector.add_id_weight(
                id_working_point = self.criteria["electron"][self.lepton_flavor]["id"]
            )

            # add electron reco weights
            electron_corrector.add_reco_weight("Above")
            electron_corrector.add_reco_weight("Below")

            # add trigger weights
            if self.lepton_flavor == "ele":
                pass
            
            # muon corrector
            if (
                self.criteria["muon"][self.lepton_flavor]["id"] == "highpt"
            ):
                mu_corrector = MuonHighPtCorrector
            else:
                mu_corrector = MuonCorrector

                
            muon_corrector = mu_corrector(
                muons=objects["muons"],
                weights=weights_container,
                year=self.year,
                variation=self.syst,
                id_wp = self.criteria["muon"][self.lepton_flavor]["id"],
                iso_wp = self.criteria["muon"][self.lepton_flavor]["iso"],
            )

            # add muon RECO weights
            muon_corrector.add_reco_weight()
            # add muon ID weights
            muon_corrector.add_id_weight()
            # add muon iso weights
            muon_corrector.add_iso_weight()

            
            # add tau weights
            tau_corrector = TauCorrector(
                taus=objects["taus"],
                weights=weights_container,
                year=self.year,
                tau_vs_jet = self.criteria["tau"][self.lepton_flavor]["fake_VSjet_pass"],
                tau_vs_ele = self.criteria["tau"][self.lepton_flavor]["fake_VSe"],
                tau_vs_mu = self.criteria["tau"][self.lepton_flavor]["fake_VSmu"],
                variation=self.syst,
            )
            tau_corrector.add_id_weight_DeepTau2017v2p1VSe()
            tau_corrector.add_id_weight_DeepTau2017v2p1VSmu()
            tau_corrector.add_id_weight_DeepTau2017v2p1VSjet()


            add_QCD_vs_Top_weight(
                    fatjets = objects["fatjets"],
                    weights = weights_container,
                    year=self.year,
                    working_point_fatjet = self.criteria["fatjet"][self.lepton_flavor]["particleNet_Top_Nominal"],                    
                    variation=self.syst
            )

            add_QCD_vs_W_weight(
                    wjets = objects["wjets"],
                    weights = weights_container,
                    year=self.year,
                    working_point_wjet = self.criteria["wjet"][self.lepton_flavor]["particleNet_W_Nominal"],
                    variation=self.syst
            )


            output["metadata"].update({"sumw_POG": ak.sum(weights_container.weight())})

            # -------------------------
            # ttbar boost correction
            # -------------------------
            add_top_boost_corrections(
                    jets = objects["jets"],
                    bjets = objects["bjets"],
                    fatjets = objects["fatjets"],
                    wjets= objects["wjets"],
                    muons = objects["muons"],
                    electrons = objects["electrons"],
                    taus = objects["taus"],
                    met = objects["met"],
                    lepton_flavor = self.lepton_flavor,
                    dataset = dataset,
                    weights = weights_container,
                    year = self.year,
                    variation = self.syst
            ) 

            # -------------------------
            # ISR correction
            # -------------------------
            ISR_weight(events=events, 
                        jets=objects["jets"], 
                        dataset=dataset, 
                        weights=weights_container, 
                        year=self.year, 
                        channel="", 
                        variation=self.syst
            )

            output["metadata"].update({"sumw_POG_plus_no_trigger": ak.sum(weights_container.weight())})


        # =============================================================================
        #                    Event selection
        # =============================================================================
        # make a PackedSelection object to store selection masks
        self.selections = PackedSelection(dtype='uint64')


        # -------------------------
        #  Luminosity
        # -------------------------
        # add luminosity calibration mask (only to data)
        lumi_mask = get_lumi_mask(events = events , year = self.year, is_mc = self.is_mc)
        self.selections.add("lumi", lumi_mask)

        # --------------------------
        #  MET filters
        # -------------------------
        met_filters_mask =  get_met_filters_mask(events = events, year = self.year, is_mc = self.is_mc)
        self.selections.add("metfilters", met_filters_mask)


        # --------------------------
        #  Good vertex
        # -------------------------
        good_vertex_mask = get_good_vertex_mask(events = events)
        self.selections.add("goodvertex", good_vertex_mask)

        # -------------------------------
        #        Trigger: OR 
        # -------------------------------
        trigger_option =  self.criteria["trigger"][self.lepton_flavor]["general"]

        trigger_mask, trigger_names = get_trigger_mask(events=events,
                                                        lepton_flavor = self.lepton_flavor,
                                                        year = self.year,
                                                        reference_trigger = trigger_option,
                                                        muon_id = self.criteria["muon"][self.lepton_flavor]["id"],
                                                        electron_id = self.criteria["electron"][self.lepton_flavor]["id"]
                                                       )

        output["metadata"].update({"Triggers": trigger_names})
        self.selections.add(f"trigger", trigger_mask)

        if self.lepton_flavor == "tau" and self.is_mc:
            # add met trigger weights: PFMETNoMu120 trigger
            add_met_trigger_corrections(
                trigger_mask, 
                dataset, 
                events.MET,
                weights_container, 
                self.year, 
                self.syst
            ) 

        # -------------------------------
        #       Trigger Match: OR 
        #   Only to Muons and Electrons
        # -------------------------------
        trigger_mask, trigger_match_mask = get_trigger_match_mask(
            events=events,
            leptons = {"mu": muons, "ele": electrons},
            lepton_flavor=self.lepton_flavor,
            year=self.year,
            electron_id_wp = self.criteria["electron"][self.lepton_flavor]["id"],
            muon_id_wp = self.criteria["muon"][self.lepton_flavor]["id"]
        )

        # add trigger weights
        if self.lepton_flavor == "mu" and self.is_mc:
            muon_corrector.add_triggeriso_weight(
                trigger_mask=trigger_mask,
                trigger_match_mask=trigger_match_mask,
            )

        # --------------------------
        #  Number of leptons and jets
        # -------------------------
        # add number of leptons and jets
        self.selections.add("one_electron", ak.num(objects["electrons"]) == 1)
        self.selections.add("electron_veto", ak.num(objects["electrons"]) == 0)

        self.selections.add("one_muon", ak.num(objects["muons"]) == 1)
        self.selections.add("muon_veto", ak.num(objects["muons"]) == 0)

        self.selections.add("one_tau", ak.num(objects["taus"]) == 1)            
        self.selections.add("tau_veto", ak.num(objects["taus"]) == 0)

        self.selections.add("one_bjet", ak.num(bjets) == 1)
        self.selections.add("bjet_veto", ak.num(bjets) == 0)


        self.selections.add(f"met",  good_met_masks)

        # --------------------------
        #     HEM cleaning
        # -------------------------        
        HEM_cleaning_mask = get_HEM_cleaning(events = events, 
                    jets = jets_veto,
                    electrons = objects["electrons"],
                    year = self.year
        )
        self.selections.add("HEMCleaning", HEM_cleaning_mask)

        # --------------------------
        #     Stitiching  
        # -------------------------
        stitching_mask = get_stitching_mask(events= events, dataset_name=dataset)        
        self.selections.add("Stitching", stitching_mask)

        # --------------------------
        # deltaphi_cut cut: 
        # --------------------------     
        delta_phi_jet_met_mask = select_good_delta_phi_jet_met(events = events, 
                                                                jets = objects["jets"], 
                                                                delta_phi_cut = self.criteria["met"][self.lepton_flavor]["delta_phi_jets_met"],
                                                                invert_delta_phi_cut = self.criteria["met"][self.lepton_flavor]["invert_delta_phi"]
                                )

        self.selections.add(f"delta_phi_jet_met", delta_phi_jet_met_mask)

        # ====================================================
        #     Define selection regions for each channel
        # ===================================================
        region_selection = {
            "tau": [
                "goodvertex",
                "Stitching",
                "lumi",
                f"trigger",
                "metfilters",
                "HEMCleaning",
                "electron_veto",
                "muon_veto",
                "one_tau",
                "one_bjet",
                f"delta_phi_jet_met",
                f"met",
            ],
            "mu": [
                "goodvertex",
                "Stitching",
                "lumi",
                f"trigger",
                "trigger_match",
                "metfilters",
                "HEMCleaning",
                "electron_veto",
                "tau_veto",
                "one_muon",
                "one_bjet",
                f"delta_phi_jet_met",
                f"met",
            ],
        }

       # --------------
        # save cutflow 
        # --------------
        cut_names = region_selection[self.lepton_flavor]
        output["metadata"].update({"cutflow": {}})
        output["metadata"].update({"cutflow_raw": {}})
        fill_cutflow(metadata = output["metadata"], cut_name = "sumw", table_name = "cutflow", weights = weights_container.weight())
        selections = []        
        for cut_name in cut_names:
            selections.append(cut_name)
            current_selection = self.selections.all(*selections)
            fill_cutflow(metadata = output["metadata"], cut_name = cut_name, table_name = "cutflow", weights = weights_container.weight()[current_selection])

        # ----------------------------
        # Save weights statistics
        # ----------------------------    
        # save sum of weights before selections
        output["metadata"].update({"sumw": ak.sum(weights_container.weight())})
        # save weights statistics
        output["metadata"].update({"weight_statistics": {}})
        for weight, statistics in weights_container.weightStatistics.items():
            output["metadata"]["weight_statistics"][weight] = statistics               


          
        # =======================================================
        #            Nominal case
        # =======================================================
        self.selections.add(
            self.lepton_flavor,
            self.selections.all(
                *region_selection[self.lepton_flavor]
            ),
        )
        region_selection_mask = self.selections.all(self.lepton_flavor)
        # check that there are events left after selection
        nevents_after = ak.sum(region_selection_mask)

        region_selection_weights = weights_container.weight()[region_selection_mask]

        

        if nevents_after == 0:
            fill_cutflow(metadata = output["metadata"], cut_name = "fail_top_tagger", table_name = "cutflow", weights = region_selection_weights)
            
            # save weighted events to metadata
            output["metadata"].update({
                "weighted_final_nevents": ak.sum(region_selection_weights),
                "raw_final_nevents": nevents_after,
            })

            nevents_top_tagger = nevents_after

        else:            
            # =============================================================
            #                   Top tagger mask
            # =============================================================
            mask_top, masks, njets_no_top, tops, selected_objects = get_topXfinder_masks(
                lepton_flavor = self.lepton_flavor,
                region_mask = region_selection_mask,
                objects = objects,
                top_tagger_cases = self.criteria["top_tagger"][self.lepton_flavor]["cases"],
                cross_cleaning = cc,
                invert_topXfinder = self.criteria["top_tagger"][self.lepton_flavor]["invert_top_tagger"]
            )

            weights = region_selection_weights[mask_top]

            fill_cutflow(metadata = output["metadata"], cut_name = "fail_top_tagger", table_name = "cutflow", weights = weights)

            nevents_top_tagger = ak.sum(mask_top)

            output_metadata(output=output["metadata"], weights=region_selection_weights, masks=masks, mask_top=mask_top)
            
            # save weighted events to metadata
            output["metadata"].update({
                "weighted_final_nevents": ak.sum(region_selection_weights[mask_top]),
                "raw_final_nevents": nevents_top_tagger,
            })
        
            if nevents_top_tagger > 0:
                # =============================================================
                #                   Filling the histograms
                # =============================================================
                histograms_output_array(
                    self_main = self,
                    lepton_flavor =  self.lepton_flavor,
                    njets_no_top = njets_no_top,
                    tops = tops,
                    objects = selected_objects,
                    mask = mask_top,
                    name = "main",
                )
        # ============================================================== 
        #                     Systematics variations
        # ============================================================== 

        if self.run_systematics and self.is_mc:

            syst_var = Systematic_variation(
                lepton_flavor = self.lepton_flavor, 
                cut_names = cut_names,  
                criteria = self.criteria,
                selections = self.selections,    
                weights_container = weights_container,                            
                metadata = output["metadata"], 
                objects = objects,
                object_variations = objects_variations, 
                delta_list = delta_list, 
                processor = self.processor
            )

            syst_var.get_syst_variation_mask(self_main = self, table_name = "cutflow", nworkers = self.criteria["top_tagger"][self.lepton_flavor]["nworkers"], name = "main")

            if nevents_top_tagger > 0:
                syst_var.get_syst_variation_event_level(name = "main", region_mask = region_selection_mask, mask = mask_top, self_main = self)

        # ====================================================
        #      QCD data-driven
        # ====================================================

        # Complementary control regions are defined based on tau and delta_phi.
        if self.qcd_data_driven:  

            QCD_ABCD = QCD_data_driven(
                self_main = self,
                year = self.year,
                lepton_flavor = self.lepton_flavor,
                is_mc = self.is_mc,
                syst = self.syst,
                criteria = self.criteria,
                events = events,
                objects = objects,
                object_variations = objects_variations, 
                weights_container = weights_container,
                selections = self.selections,
                region_selection_list = cut_names,
                output_metadata = output["metadata"],
                delta_list = delta_list if self.is_mc else None, 
                processor = self.processor
            )
            
            QCD_ABCD.get_region_selection_BCD(run_systematics = self.run_systematics, is_mc= self.is_mc)

             

        """
        if nevents_top_tagger > 0:
            # =============================================================
            #                   Filling the histograms
            # =============================================================
            histograms_output_syst(self, 
                            njets_no_top = njets_no_top,
                            bjets = selected_objects["bjets"], 
                            jets = selected_objects["jets"],
                            fatjets = selected_objects["fatjets"], 
                            wjets = selected_objects["wjets"],
                            electrons = selected_objects["electrons"], 
                            muons = selected_objects["muons"],
                            taus = selected_objects["taus"], 
                            met = selected_objects["met"],
                            tops = tops , 
                            mask = mask_top, 
                            lepton_flavor = self.lepton_flavor, 
                            is_mc = self.is_mc, 
                            events = selected_objects["events"],
                            syst_flag = "nominal")

        """

        # define output dictionary accumulator
        if self.output_type == "array":

            array_dict = {}
            self.add_feature(
                "weights_main", region_selection_weights[mask_top]
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

            output["arrays"] = array_dict


        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator