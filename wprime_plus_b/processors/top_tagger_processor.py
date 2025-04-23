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
from wprime_plus_b.corrections.met import apply_met_phi_corrections, add_met_trigger_corrections, update_met_jet_veto, met_noMu_cal, met_recoil, met_noMu_minus, met_noMu_plus
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
from wprime_plus_b.corrections.ISR import ISR_weight
from wprime_plus_b.corrections.ttbar_boost import add_ttbar_boost_corrections

# Selections: Config
from wprime_plus_b.selections.top_tagger.bjet_config import top_tagger_bjet_selection
from wprime_plus_b.selections.top_tagger.cases_top_tagger_config import top_tagger_cases_selection, top_tagger_mW_mTop_Njets_selection
from wprime_plus_b.selections.top_tagger.electron_config import top_tagger_electron_selection
from wprime_plus_b.selections.top_tagger.fatjet_config import top_tagger_fatjet_selection
from wprime_plus_b.selections.top_tagger.general_config import top_tagger_cross_cleaning_selection, top_tagger_trigger_selection
from wprime_plus_b.selections.top_tagger.jet_config import top_tagger_jet_selection
from wprime_plus_b.selections.top_tagger.met_config import top_tagger_met_selection
from wprime_plus_b.selections.top_tagger.muon_config import top_tagger_muon_selection
from wprime_plus_b.selections.top_tagger.tau_config import top_tagger_tau_selection
from wprime_plus_b.selections.top_tagger.wjet_config import top_tagger_wjet_selection

# Selections: objects
from wprime_plus_b.selections.top_tagger.bjet_selection import select_good_bjets
from wprime_plus_b.selections.top_tagger.electron_selection import select_good_electrons
from wprime_plus_b.selections.top_tagger.fatjet_selection import select_good_fatjets
from wprime_plus_b.selections.top_tagger.jet_selection import select_good_jets
from wprime_plus_b.selections.top_tagger.muon_selection import select_good_muons
from wprime_plus_b.selections.top_tagger.tau_selection import select_good_taus
from wprime_plus_b.selections.top_tagger.wjet_selection import select_good_wjets

# Top tagger
from wprime_plus_b.processors.utils.topXfinder import topXfinder



from wprime_plus_b.processors.utils.analysis_utils import delta_r_mask, normalize, trigger_match, top_tagger, output_metadata, histograms_output


class TopTaggerProccessor(processor.ProcessorABC):
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
        run_systematics: str = "false",
    ):
        self.run_systematics = (run_systematics.lower() == "true")
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
            "bjet_kin": histograms.ttbar_bjet_hist,
            "met_kin": histograms.ttbar_met_hist,
            "lepton_kin": histograms.ttbar_lepton_hist,
            "lepton_bjet_kin": histograms.ttbar_lepton_bjet_hist,
            "lepton_met_kin": histograms.ttbar_lepton_met_hist,
            "lepton_met_bjet_kin": histograms.ttbar_lepton_met_bjet_hist,
            "top_mrec": histograms.top_tagger_hist,
            "ST_HT": histograms.st_ht_hist,
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

                
                # Apply corrections only if there are jets present in at least one event
                if ak.any(ak.num(events.FatJet) > 0):
                    # fatjets
                    apply_fatjet_corrections(events, self.year)
                
                
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
                    "ele": top_tagger_electron_selection[self.lepton_flavor]["electron_id_wp"],
                    "mu": top_tagger_muon_selection[self.lepton_flavor]["muon_id_wp"]
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
                trigger_match_mask = np.ones(len(events), dtype="bool")

            # -------------------------------------------------------------
            # Veto Jets
            # -------------------------------------------------------------
            jet_veto_mask = jetvetomaps_mask(events.Jet, self.year, "jetvetomap")
            jets_veto = events.Jet[jet_veto_mask]


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
                add_l1prefiring_weight(events, weights_container, self.year, syst_var)
                # add pileup weigths
                add_pileup_weight(events, weights_container, self.year, syst_var)

                # add pujetid weigths
                add_pujetid_weight(
                    jets=jets_veto,
                    weights=weights_container,
                    year=self.year,
                    working_point=top_tagger_bjet_selection[self.lepton_flavor][
                        "bjet_pileup_id"
                    ],
                    variation=syst_var,
                )
                """
                # b-tagging corrector
                btag_corrector = BTagCorrector(
                    jets=jets_veto,
                    weights=weights_container,
                    sf_type="comb",
                    worging_point=top_tagger_bjet_selection[self.lepton_flavor][
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
                    id_working_point=top_tagger_electron_selection[
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
                    top_tagger_muon_selection[self.lepton_flavor]["muon_id_wp"]
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
                    id_wp=top_tagger_muon_selection[self.lepton_flavor][
                        "muon_id_wp"
                    ],
                    iso_wp=top_tagger_muon_selection[self.lepton_flavor][
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
                    tau_vs_jet=top_tagger_tau_selection[self.lepton_flavor][
                        "tau_vs_jet"
                    ],
                    tau_vs_ele=top_tagger_tau_selection[self.lepton_flavor][
                        "tau_vs_ele"
                    ],
                    tau_vs_mu=top_tagger_tau_selection[self.lepton_flavor][
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
                        working_point_fatjet = top_tagger_fatjet_selection[self.lepton_flavor]["TvsQCD"],
                        variation=syst_var
                )

                add_QCD_vs_W_weight(
                        wjets = events.FatJet,
                        weights = weights_container,
                        year=self.year,
                        year_mod="",
                        working_point_wjet = top_tagger_wjet_selection[self.lepton_flavor]["WvsQCD"],
                        variation=syst_var
                )
                """
                
                if self.lepton_flavor == "tau":
                    # add met trigger SF
                    add_met_trigger_corrections(trigger_mask, dataset, events.MET, weights_container, self.year, "", syst_var)                    
                
                    
                    
            # -------------------------------------------------------------
            # object selection
            # -------------------------------------------------------------

            # Cross_cleaning:
            cc = top_tagger_cross_cleaning_selection[self.lepton_flavor]["DR"]
            
            # select good electrons
            good_electrons = select_good_electrons(
                events=events,
                electron_pt_threshold=top_tagger_electron_selection[
                    self.lepton_flavor
                ]["electron_pt_threshold"],
                electron_eta_threshold = top_tagger_electron_selection[
                    self.lepton_flavor
                ]["electron_eta_threshold"],
                electron_id_wp=top_tagger_electron_selection[
                    self.lepton_flavor
                ]["electron_id_wp"],
                electron_iso_wp=top_tagger_electron_selection[
                    self.lepton_flavor
                ]["electron_iso_wp"],
            )
            electrons = events.Electron[good_electrons]



            # select good muons
            good_muons_masks = select_good_muons(
                events=events,
                muon_pt_threshold=top_tagger_muon_selection[
                    self.lepton_flavor
                ]["muon_pt_threshold"],
                muon_eta_threshold = top_tagger_muon_selection[
                    self.lepton_flavor
                ]["muon_eta_threshold"],
                muon_id_wp= top_tagger_muon_selection[
                    self.lepton_flavor
                ]["muon_id_wp"],
                muon_iso_wp=top_tagger_muon_selection[
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
                tau_pt_threshold=top_tagger_tau_selection[
                    self.lepton_flavor
                ]["tau_pt_threshold"],
                tau_eta_threshold=top_tagger_tau_selection[
                    self.lepton_flavor
                ]["tau_eta_threshold"],
                tau_dz_threshold=top_tagger_tau_selection[
                    self.lepton_flavor
                ]["tau_dz_threshold"],
                tau_vs_jet=top_tagger_tau_selection[
                    self.lepton_flavor
                ]["tau_vs_jet"],
                tau_vs_ele=top_tagger_tau_selection[
                    self.lepton_flavor
                ]["tau_vs_ele"],
                tau_vs_mu=top_tagger_tau_selection[
                    self.lepton_flavor
                ]["tau_vs_mu"],
                prong=top_tagger_tau_selection[
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
                jets=jets_veto ,
                year=self.year,
                btag_working_point=top_tagger_bjet_selection[
                    self.lepton_flavor
                ]["btag_working_point"],
                jet_pt_threshold=top_tagger_bjet_selection[
                    self.lepton_flavor
                ]["bjet_pt_threshold"],
                jet_eta_threshold = top_tagger_bjet_selection[
                    self.lepton_flavor
                ]["bjet_eta_threshold"],
                jet_id_wp=top_tagger_bjet_selection[
                    self.lepton_flavor
                ]["bjet_id_wp"],
                jet_pileup_id=top_tagger_bjet_selection[
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
                btag_working_point=top_tagger_jet_selection[
                    self.lepton_flavor
                ]["fail_btag_working_point"],
                jet_pt_threshold=top_tagger_jet_selection[
                    self.lepton_flavor
                ]["jet_pt_threshold"],
                jet_eta_threshold = top_tagger_jet_selection[
                    self.lepton_flavor
                ]["jet_eta_threshold"],
                jet_id_wp=top_tagger_jet_selection[
                    self.lepton_flavor
                ]["jet_id_wp"],
                jet_pileup_id=top_tagger_jet_selection[
                    self.lepton_flavor
                ]["jet_pileup_id"],
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
                fatjet_pt_threshold = top_tagger_fatjet_selection[
                    self.lepton_flavor
                ]["fatjet_pt_threshold"],
                fatjet_eta_threshold = top_tagger_fatjet_selection[
                    self.lepton_flavor
                ]["fatjet_eta_threshold"],
                TvsQCD = top_tagger_fatjet_selection[
                    self.lepton_flavor
                ]["TvsQCD"],
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
                w_pt_threshold = top_tagger_wjet_selection[
                    self.lepton_flavor
                ]["wjet_pt_threshold"],
                w_eta_threshold = top_tagger_wjet_selection[
                    self.lepton_flavor
                ]["wjet_eta_threshold"],
                WvsQCD = top_tagger_wjet_selection[
                    self.lepton_flavor
                ]["WvsQCD"],
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


            if self.run_systematics:
                if self.is_mc:
                    good_muons_up = (good_muons_masks["up"]) & (
                        delta_r_mask(events.Muon, electrons, threshold=cc)
                    )
                    good_muons_down = (good_muons_masks["down"]) & (
                        delta_r_mask(events.Muon, electrons, threshold=cc)
                    )
                    muons_up = events.Muon[good_muons_up]
                    muons_down = events.Muon[good_muons_down]
                


                    good_taus_up = (
                        (good_taus_masks["up"])
                        & (delta_r_mask(events.Tau, electrons, threshold=cc))
                        & (delta_r_mask(events.Tau, muons, threshold=cc))
                    )
                    good_taus_down = (
                        (good_taus_masks["down"])
                        & (delta_r_mask(events.Tau, electrons, threshold=cc))
                        & (delta_r_mask(events.Tau, muons, threshold=cc))
                    )
                    taus_up = events.Tau[good_taus_up]
                    taus_down = events.Tau[good_taus_down]




                    good_bjets_jes_up = (
                        good_bjets_masks["JES_up"]
                        & (delta_r_mask(jets_veto, electrons, threshold=cc))
                        & (delta_r_mask(jets_veto, muons, threshold=cc))
                        & (delta_r_mask(jets_veto, taus, threshold=cc))
                    )
                    good_bjets_jes_down = (
                        good_bjets_masks["JES_down"]
                        & (delta_r_mask(jets_veto, electrons, threshold=cc))
                        & (delta_r_mask(jets_veto, muons, threshold=cc))
                        & (delta_r_mask(jets_veto, taus, threshold=cc))
                    )
                    good_bjets_jer_up = (
                        good_bjets_masks["JER_up"]
                        & (delta_r_mask(jets_veto, electrons, threshold=cc))
                        & (delta_r_mask(jets_veto, muons, threshold=cc))
                        & (delta_r_mask(jets_veto, taus, threshold=cc))
                    )
                    good_bjets_jer_down = (
                        good_bjets_masks["JER_down"]
                        & (delta_r_mask(jets_veto, electrons, threshold=cc))
                        & (delta_r_mask(jets_veto, muons, threshold=cc))
                        & (delta_r_mask(jets_veto, taus, threshold=cc))
                    )
                    bjets_jes_up = jets_veto[good_bjets_jes_up]
                    bjets_jes_down = jets_veto[good_bjets_jes_down]
                    bjets_jer_down = jets_veto[good_bjets_jer_down]
                    bjets_jer_up = jets_veto[good_bjets_jer_up]



                    good_jets_jes_up = (
                        good_jets_masks["JES_up"]
                        & (delta_r_mask(jets_veto, electrons, threshold=cc))
                        & (delta_r_mask(jets_veto, muons, threshold=cc))
                        & (delta_r_mask(jets_veto, taus, threshold=cc))
                    )
                    good_jets_jes_down = (
                        good_jets_masks["JES_down"]
                        & (delta_r_mask(jets_veto, electrons, threshold=cc))
                        & (delta_r_mask(jets_veto, muons, threshold=cc))
                        & (delta_r_mask(jets_veto, taus, threshold=cc))
                    )
                    good_jets_jer_up = (
                        good_jets_masks["JER_up"]
                        & (delta_r_mask(jets_veto, electrons, threshold=cc))
                        & (delta_r_mask(jets_veto, muons, threshold=cc))
                        & (delta_r_mask(jets_veto, taus, threshold=cc))
                    )
                    good_jets_jer_down = (
                        good_jets_masks["JER_down"]
                        & (delta_r_mask(jets_veto, electrons, threshold=cc))
                        & (delta_r_mask(jets_veto, muons, threshold=cc))
                        & (delta_r_mask(jets_veto, taus, threshold=cc))
                    )
                    jets_jes_up = jets_veto[good_jets_jes_up]
                    jets_jes_down = jets_veto[good_jets_jes_down]
                    jets_jer_up = jets_veto[good_jets_jer_up]
                    jets_jer_down = jets_veto[good_jets_jer_down]



                    good_fatjets_jes_up = (
                        good_fatjets_masks["JES_up"]
                        & (delta_r_mask(events.FatJet, electrons, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, muons, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, taus, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, bjets, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, jets, threshold = 2*cc))
                    )   
                    good_fatjets_jes_down = (
                        good_fatjets_masks["JES_down"]
                        & (delta_r_mask(events.FatJet, electrons, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, muons, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, taus, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, bjets, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, jets, threshold = 2*cc))
                    )   
                    good_fatjets_jer_up = (
                        good_fatjets_masks["JER_up"]
                        & (delta_r_mask(events.FatJet, electrons, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, muons, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, taus, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, bjets, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, jets, threshold = 2*cc))
                    )   
                    good_fatjets_jer_down = (
                        good_fatjets_masks["JER_down"]
                        & (delta_r_mask(events.FatJet, electrons, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, muons, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, taus, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, bjets, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, jets, threshold = 2*cc))
                    )   
                    fatjets_jes_up = events.FatJet[good_fatjets_jes_up]
                    fatjets_jes_down = events.FatJet[good_fatjets_jes_down]
                    fatjets_jer_up = events.FatJet[good_fatjets_jer_up]
                    fatjets_jer_down = events.FatJet[good_fatjets_jer_down]


                    good_wjets_jes_up = (
                        good_wjets_masks["JES_up"]
                        & (delta_r_mask(events.FatJet, electrons, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, muons, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, taus, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, bjets, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, jets, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, fatjets, threshold = 2*cc))
                    )   
                    good_wjets_jes_down = (
                        good_wjets_masks["JES_down"]
                        & (delta_r_mask(events.FatJet, electrons, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, muons, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, taus, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, bjets, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, jets, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, fatjets, threshold = 2*cc))
                    )   
                    good_wjets_jer_up = (
                        good_wjets_masks["JER_up"]
                        & (delta_r_mask(events.FatJet, electrons, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, muons, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, taus, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, bjets, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, jets, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, fatjets, threshold = 2*cc))
                    )  
                    good_wjets_jer_down = (
                        good_wjets_masks["JER_down"]
                        & (delta_r_mask(events.FatJet, electrons, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, muons, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, taus, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, bjets, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, jets, threshold = 2*cc))
                        & (delta_r_mask(events.FatJet, fatjets, threshold = 2*cc))
                    )   
                    wjets_jes_up = events.FatJet[good_wjets_jes_up]
                    wjets_jes_down = events.FatJet[good_wjets_jes_down] 
                    wjets_jer_up = events.FatJet[good_wjets_jer_up]
                    wjets_jer_down = events.FatJet[good_wjets_jer_down]

            # New weights:
            if self.is_mc:
                # b-tagging corrector
                btag_corrector = BTagCorrector(
                    jets=bjets,
                    weights=weights_container,
                    sf_type="comb",
                    worging_point=top_tagger_bjet_selection[self.lepton_flavor][
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
                    electrons=electrons,
                    weights=weights_container,
                    year=self.year,
                )

                # add electron ID weights
                electron_corrector.add_id_weight(
                    id_working_point=top_tagger_electron_selection[
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
                    top_tagger_muon_selection[self.lepton_flavor]["muon_id_wp"]
                    == "highpt"
                ):
                    mu_corrector = MuonHighPtCorrector
                else:
                    mu_corrector = MuonCorrector
                muon_corrector = mu_corrector(
                    muons=muons,
                    weights=weights_container,
                    year=self.year,
                    variation=syst_var,
                    id_wp=top_tagger_muon_selection[self.lepton_flavor][
                        "muon_id_wp"
                    ],
                    iso_wp=top_tagger_muon_selection[self.lepton_flavor][
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
                    tau_vs_jet=top_tagger_tau_selection[self.lepton_flavor][
                        "tau_vs_jet"
                    ],
                    tau_vs_ele=top_tagger_tau_selection[self.lepton_flavor][
                        "tau_vs_ele"
                    ],
                    tau_vs_mu=top_tagger_tau_selection[self.lepton_flavor][
                        "tau_vs_mu"
                    ],
                    variation=syst_var,
                )
                tau_corrector.add_id_weight_DeepTau2017v2p1VSe()
                tau_corrector.add_id_weight_DeepTau2017v2p1VSmu()
                tau_corrector.add_id_weight_DeepTau2017v2p1VSjet()


                add_QCD_vs_Top_weight(
                        fatjets = fatjets,
                        weights = weights_container,
                        year=self.year,
                        year_mod="",
                        working_point_fatjet = top_tagger_fatjet_selection[self.lepton_flavor]["TvsQCD"],
                        variation=syst_var
                )

                add_QCD_vs_W_weight(
                        wjets = wjets,
                        weights = weights_container,
                        year=self.year,
                        year_mod="",
                        working_point_wjet = top_tagger_wjet_selection[self.lepton_flavor]["WvsQCD"],
                        variation=syst_var
                )

            # -------------------------
            # p_T^{miss} correction
            # -------------------------
            update_met_jet_veto(events = events, jets_veto = jets_veto)  
            met_noMu_cal(events = events, muons = muons)
            met_noMu_plus(events = events, muons = muons)
            met_noMu_minus(events = events, muons = muons)
            met_recoil(events = events, muons = muons, electrons = electrons, taus = taus)

            # -------------------------
            # ST correction
            # -------------------------
            weights_copy = copy.deepcopy(weights_container)

            add_ttbar_boost_corrections(
                    jets = jets,
                    bjets = bjets,
                    muons = muons,
                    electrons = electrons,
                    taus = taus,
                    met = events.MET,
                    lepton_flavor = self.lepton_flavor,
                    dataset = dataset,
                    weights = weights_container,
                    year = self.year,
                    variation = syst_var,
            ) 



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

            # check that there be a minimum MET greater than the threshold

            met_threshold =  top_tagger_met_selection[self.lepton_flavor]["met_threshold"]
            self.selections.add(f"met_{met_threshold}",  events.MET.pt > met_threshold)

            if self.is_mc:
                met_variations = {
                    "nominal": events.MET.pt,
                    "up": events.MET.MET_UnclusteredEnergy.up.pt,
                    "down": events.MET.MET_UnclusteredEnergy.down.pt,
                }
                
                """
                for variation, met_pt in met_variations.items():
                    if variation == "nominal":
                        self.selections.add(f"met_{met_threshold}", met_pt > met_threshold)
                    else:
                        self.selections.add(f"met_{met_threshold}_{variation}", met_pt > met_threshold)"
                """
            else:
                self.selections.add(f"met_{met_threshold}", events.MET.pt > met_threshold)


            
            self.selections.add(f"met_recoil_{met_threshold}", events.MET.pt_recoil > met_threshold)   
            
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

            # -------- Trigger: OR ----------#
            # Reference trigger
            reference_trigger =  top_tagger_trigger_selection[self.lepton_flavor]["trigger"]
            if self.lepton_flavor == "mu":
                mu_id = top_tagger_muon_selection[self.lepton_flavor]["muon_id_wp"]
                with importlib.resources.path(
                    "wprime_plus_b.data", "triggers.json"
                ) as path:
                    with open(path, "r") as handle:
                        ref_trigger = json.load(handle)[self.year][reference_trigger][mu_id]
                        print(ref_trigger, type(ref_trigger))
            
            elif self.lepton_flavor ==  "ele":
                ele_id = top_tagger_electron_selection[self.lepton_flavor]["electron_id_wp"]
                with importlib.resources.path(
                    "wprime_plus_b.data", "triggers.json"
                ) as path:
                    with open(path, "r") as handle:
                        ref_trigger = json.load(handle)[self.year][reference_trigger][ele_id]
                        print(ref_trigger, type(ref_trigger))               

            elif self.lepton_flavor ==  "tau":
                with importlib.resources.path(
                    "wprime_plus_b.data", "triggers.json"
                ) as path:
                    with open(path, "r") as handle:
                        ref_trigger = json.load(handle)[self.year][reference_trigger]
                        print(ref_trigger, type(ref_trigger))    

            #reference_triggers =  [trigger for trigger in events.HLT.fields if trigger.startswith(ref_trigger)]
            reference_triggers = [
                trigger for trigger in events.HLT.fields if any(trigger.startswith(r) for r in ref_trigger)
            ]
            
            mask_reference_trigger = np.zeros(len(events), dtype="bool")
            
            for trigger_reference in reference_triggers:
                if trigger_reference in events.HLT.fields:
                    print(f"Reference trigger: {trigger_reference}")
                    mask_reference_trigger = mask_reference_trigger | events.HLT[trigger_reference]

            self.selections.add(f"trigger_{reference_trigger}", mask_reference_trigger)


            # Trigger under study
            study_trigger_option =  top_tagger_trigger_selection[self.lepton_flavor]["trigger_eff"] 
            if study_trigger_option != "":
                with importlib.resources.path(
                    "wprime_plus_b.data", "triggers.json"
                ) as path:
                    with open(path, "r") as handle:
                        study_trigger = json.load(handle)[self.year][study_trigger_option]
                        print(study_trigger, type(study_trigger))   

                triggers_under_study = [
                    trigger for trigger in events.HLT.fields if any(trigger.startswith(r) for r in study_trigger)
                ]           
                mask_under_study = np.zeros(len(events), dtype="bool")
                for trigger_under_study in triggers_under_study:
                    if trigger_under_study in events.HLT.fields:
                        print(f"Study trigger: {trigger_under_study}")
                        mask_under_study = mask_under_study | events.HLT[trigger_under_study]


            # --------------------------
            # deltaphi_cut cut
            # --------------------------     
            delta_phi_met_jet = jets.delta_phi(events.MET)       
            delta_phi_jet_met_pass = ak.all(np.abs(delta_phi_met_jet) > 0.7, axis=1)
            self.selections.add("delta_phi_jet_met_0.7", delta_phi_jet_met_pass)

            # define selection regions for each channel
            region_selection = {
               "tau": [
                    "goodvertex",
                    "Stitching",
                    "lumi",
                    f"trigger_{reference_trigger}",
                    "metfilters",
                    "HEMCleaning",
                    f"met_{met_threshold}",
 #                   "delta_phi_jet_met_0.7",
                    "electron_veto",
                    "muon_veto",
                    "one_tau",
                ],
                "mu": [
                    "goodvertex",
                    "Stitching",
                    "lumi",
                    "metfilters",
                    "HEMCleaning",
                    f"trigger_{reference_trigger}",
                    "trigger_match",
                    "delta_phi_jet_met_0.7",
                    f"met_{met_threshold}",
                    "electron_veto",
                    "tau_veto",
                    "one_muon",
                ],
            }

            # Crear una copia de region_selection para modificaciones posteriores con las variaciones Up/Down a nivel de objeto
            region_selection_variations =  copy.deepcopy(region_selection)


            # ----------------------------
            # Save weights statistics
            # ----------------------------    
            if syst_var == "nominal":
                # save sum of weights before selections
                output["metadata"].update({"sumw": ak.sum(weights_container.weight())})
                # save sum of weights before selections without ttbar boost
                output["metadata"].update({"sumw_no_ttboost": ak.sum(weights_copy.weight())})
                # save weights statistics
                output["metadata"].update({"weight_statistics": {}})
                for weight, statistics in weights_container.weightStatistics.items():
                    output["metadata"]["weight_statistics"][weight] = statistics



            # --------------
            # save cutflow 
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


            # Helper function to apply masks and selections
            def apply_selection(objects, mask):
                return {key: obj[mask] for key, obj in objects.items()}

            # Create a dictionary of objects to simplify handling
            objects = {
                "bjets": bjets,
                "jets": jets,
                "fatjets": fatjets,
                "wjets": wjets,
                "electrons": electrons,
                "muons": muons,
                "taus": taus,
                "met": events.MET,
                "events": events
            }

            if nevents_after == 0:
                output["metadata"]["cutflow"]["passing_top_tagger"] = ak.sum(weights_container.weight()[region_selection])
                output_metadata(output = output["metadata"])
                tops = ak.zeros_like(region_selection)

            else:

                #########################
                ######### Top tagger ####
                #########################
                # Top tagger cases
                cases = [f"case_{i}" for i in range(1, 14) if top_tagger_cases_selection[self.lepton_flavor].get(f"case_{i}", False)]
                
                # Apply region selection to objects
                selected_objects = apply_selection(objects, region_selection)
     
                topX = topXfinder(self.lepton_flavor, selected_objects["bjets"], selected_objects["jets"], selected_objects["fatjets"],
                                    selected_objects["wjets"],  cc)


                tops, mask_top, masks = top_tagger(topX, top_tagger_cases=cases)

                # Update cutflow
                output["metadata"]["cutflow"]["passing_top_tagger"] = ak.sum(weights_container.weight()[region_selection][mask_top])


                # Eff studies                
                trigger_eff =  top_tagger_trigger_selection[self.lepton_flavor]["trigger_eff"]

                if trigger_eff == "tau":
                    # Reduce object to passing top tagger events
                    selected_objects = apply_selection(selected_objects, mask_top)                    
                    tops_tmp = tops[mask_top]
                    trigger_mask = mask_under_study[region_selection][mask_top]
                    
                
                    # Apply trigger mask to weights and masks "mask per case"
                    pre_weights_tmp = weights_container.weight()[region_selection][mask_top]
                    masks_tmp = {key: mask[mask_top] for key, mask in masks.items()}

                    # Overlap results
                    pre_weights = pre_weights_tmp
                    masks = {key: ak.where(trigger_mask, mask, ak.Array([False] * len(mask))) for key, mask in masks_tmp.items()}
                    mask_top = trigger_mask
                    tops = tops_tmp

                    # Update cutflow for trigger efficiency
                    output["metadata"]["cutflow"][f"trigger_{study_trigger_option}"] = ak.sum(pre_weights[mask_top])


                else:
                    pre_weights = weights_container.weight()[region_selection]


                # Number of events after top tagger/efficiency
                nevents_top_tagger = ak.sum(mask_top)
                output_metadata(output=output["metadata"], weights=pre_weights, masks=masks, mask_top=mask_top)



                if nevents_top_tagger > 0:
                    # Histograms
                    histograms_output(self, selected_objects["bjets"], selected_objects["jets"], 
                                    selected_objects["electrons"], selected_objects["muons"], 
                                    selected_objects["taus"], selected_objects["met"], 
                                    tops, mask_top, self.lepton_flavor, self.is_mc, selected_objects["events"])


                    if syst_var == "nominal":
                        # save weighted events to metadata
                        output["metadata"].update({
                                "weighted_final_nevents": ak.sum(pre_weights[mask_top]),
                                "raw_final_nevents": nevents_top_tagger,
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
                                    region_weight = pre_weights[mask_top]
                                else:
                                    region_weight = weights_container.weight(
                                        modifier=variation
                                    )[mask_top]
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
                            region_weight = pre_weights[mask_top]
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
                            region_weight = pre_weights[mask_top]
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
                            "weights", weights_container.weight()[region_selection][mask_top] 
                        )


                        if self.is_mc == True:
                            # Agregar variaciones de peso
                            for variation_case, weights_case in weights_container._modifiers.items():
                                array_dict[f"{variation_case}"] = processor.column_accumulator(
                                    weights_case[region_selection][mask_top]
                                )
                        # Guardar pesos individuales filtrados por region_selection
                        for weight in weights_container.weightStatistics:
                            filtered_weight = weights_container.partial_weight(include=[weight])[region_selection][mask_top]
                            self.add_feature(weight, filtered_weight)
                            
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

            if self.run_systematics:
                # -------------------------------------------------
                # New block: up/down variations in objects.
                # -------------------------------------------------
                if self.is_mc:
                    
                    weights_container_variations = copy.deepcopy(weights_container)

        
                    # Crear un mapa para almacenar todas las variaciones
                    region_selection_map = {}
                    
                                                
                    self.selections.add("one_tau_up", ak.num(taus_up) == 1)
                    self.selections.add("one_tau_down", ak.num(taus_down) == 1)
                    self.selections.add(f"met_{met_threshold}_up", met_variations["up"] > met_threshold)
                    self.selections.add(f"met_{met_threshold}_down", met_variations["down"] > met_threshold)
                    self.selections.add("muon_veto_up", ak.num(muons_up) == 0)
                    self.selections.add("muon_veto_down", ak.num(muons_down) == 0)


                    # Mapa de modificaciones: clave es el campo a buscar, valor es el nuevo criterio
                    modification_map = {
                        "one_tau": {
                            "up": "one_tau_up",
                            "down": "one_tau_down",
                        },
                        f"met_{met_threshold}": {
                            "up": f"met_{met_threshold}_up",
                            "down": f"met_{met_threshold}_down",
                        },
                        "muon_veto": {
                            "up": "muon_veto_up",
                            "down": "muon_veto_down",
                        },
                        "jes":{
                            "up": "jes_up",
                            "down": "jes_down",
                        },
                        "jer":{
                            "up": "jer_up",
                            "down": "jer_down",
                        }
                    }

                    # Iterar sobre las modificaciones y crear una nueva versión de region_selection_variations para cada cambio
                    for key, variations in modification_map.items():
                        if key in ["one_tau", f"met_{met_threshold}", "muon_veto"]:
                            for variation in variations.keys():  # Solo usa las claves "up" y "down"
                                modified_region_selection = copy.deepcopy(region_selection_variations)

                                # Si el criterio original está presente, eliminarlo y agregar la versión modificada
                                if key in modified_region_selection[self.lepton_flavor]:
                                    modified_region_selection[self.lepton_flavor] = [
                                        criterion for criterion in modified_region_selection[self.lepton_flavor] if criterion != key
                                    ]

                                    modified_region_selection[self.lepton_flavor].append(f"{key}_{variation}")  # Agrega "one_tau_up", etc.


                                # Guardar la nueva variación en el mapa
                                region_selection_map[f"{key}_{variation}"] = modified_region_selection
                        else:
                            # no modificar region_selection, El cambio solo afectará el top tagger
                            for variation in variations.keys():
                                region_selection_map[f"{key}_{variation}"] = copy.deepcopy(region_selection_variations)


                    # Crear un nuevo objeto PackedSelection para las variaciones
                    #self.variations_selections = PackedSelection()

                    # Crear máscaras para cada variación en region_selection_map
                    for variation_name, selection in region_selection_map.items():

                        # Crear un nuevo objeto PackedSelection para esta iteración
                        temp_selections = PackedSelection()

                        # Agregar la selección temporalmente
                        temp_selections.add(
                            f"{self.region}_{variation_name}",
                            self.selections.all(
                                *selection[self.lepton_flavor]
                            ),
                        )

                        region_selection_tmp = temp_selections.all(f"{self.region}_{variation_name}")



                        # check that there are events left after selection
                        nevents_after_tmp = ak.sum(region_selection_tmp)

                        if nevents_after_tmp == 0:
                            mask_top_tmp = ak.zeros_like(region_selection_tmp, dtype=bool)  # Crear una máscara con Falses


                            # Agregar lepton_met_mass_tmp a array_dict
                            array_dict[f"lepton_met_mass_{variation_name}"] = processor.column_accumulator(np.array([]))
                            # Convertir weights_{variation_name} a un array de NumPy antes de guardarlo
                            array_dict[f"weights_{variation_name}"] = processor.column_accumulator(weights_container_variations.weight()[region_selection_tmp])

    

                        else:
                            #########################
                            ######### Top tagger ####
                            #########################
                            # Create a dictionary of objects to simplify handling
                            objects_tmp = {
                                "bjets": bjets_jes_up if "jes_up" in variation_name else
                                        bjets_jes_down if "jes_down" in variation_name else
                                        bjets_jer_up if "jer_up" in variation_name else
                                        bjets_jer_down if "jes_down" in variation_name else
                                        bjets,
                                "jets": jets_jes_up if "jes_up" in variation_name else
                                        jets_jes_down if "jes_down" in variation_name else
                                        jets_jer_up if "jer_up" in variation_name else
                                        jets_jer_down if "jer_down" in variation_name else
                                        jets,
                                "fatjets": fatjets_jes_up if "jes_up" in variation_name else
                                        fatjets_jes_down if "jes_down" in variation_name else
                                        fatjets_jer_up if "jet_up" in variation_name else
                                        fatjets_jer_down if "jer_down" in variation_name else
                                        fatjets,
                                "wjets": wjets_jes_up if "jes_up" in variation_name else
                                        wjets_jes_down if "jes_down" in variation_name else
                                        wjets_jer_up if "jer_up" in variation_name else
                                        wjets_jer_down if "jer_down" in variation_name else
                                        wjets,
                                "electrons": electrons,
                                "muons": muons_up if "muon_up" in variation_name else
                                        muons_down if "muon_down" in variation_name else
                                        muons,
                                "taus": taus_up if "one_tau_up" in variation_name else
                                        taus_down if "one_tau_down" in variation_name else
                                        taus,
                                "met": met_variations["up"] if "met_up" in variation_name else
                                    met_variations["down"] if "met_down" in variation_name else
                                    events.MET,
                                "events": events,
                            }

                            # Top tagger cases
                            cases_tmp = [f"case_{i}" for i in range(1, 14) if top_tagger_cases_selection[self.lepton_flavor].get(f"case_{i}", False)]

                            
                            # Apply region selection to objects
                            selected_objects_tmp = apply_selection(objects_tmp, region_selection_tmp)
        
                            topX_tmp = topXfinder(self.lepton_flavor, selected_objects_tmp["bjets"], selected_objects_tmp["jets"], selected_objects_tmp["fatjets"],
                                                selected_objects_tmp["wjets"],  cc)


                            tops_tmp, mask_top_tmp, masks_tmp = top_tagger(topX_tmp, top_tagger_cases=cases_tmp)

                            

                            # Aplicar las máscaras region_selection_tmp y mask_top_tmp a los objetos
                            filtered_objects_tmp = {
                                key: obj[mask_top_tmp] for key, obj in selected_objects_tmp.items()
                            }

                            if ak.sum(mask_top_tmp) > 0:
                                # Definir el mapa de leptones según el flavor
                                lepton_region_map = {
                                    "ele": filtered_objects_tmp["electrons"],
                                    "mu": filtered_objects_tmp["muons"],
                                    "tau": filtered_objects_tmp["taus"],
                                }

                                # Calcular lepton_met_mass con los objetos filtrados
                                region_leptons_tmp = lepton_region_map[self.lepton_flavor]
                                region_met_tmp = filtered_objects_tmp["met"]

                                lepton_met_mass_tmp = np.sqrt(
                                    2.0
                                    * region_leptons_tmp.pt
                                    * region_met_tmp.pt
                                    * (
                                        ak.ones_like(region_met_tmp.pt)
                                        - np.cos(region_leptons_tmp.delta_phi(region_met_tmp))
                                    )
                                )

                                # Convertir lepton_met_mass_tmp a un array de NumPy
                                lepton_met_mass_tmp_numpy = ak.to_numpy(ak.flatten(lepton_met_mass_tmp))

                                # Agregar lepton_met_mass_tmp a array_dict
                                array_dict[f"lepton_met_mass_{variation_name}"] = processor.column_accumulator(lepton_met_mass_tmp_numpy)

                            # Convertir weights_{variation_name} a un array de NumPy antes de guardarlo
                            array_dict[f"weights_{variation_name}"] = processor.column_accumulator(weights_container_variations.weight()[region_selection_tmp][mask_top_tmp])
                            
            
            output["arrays"] = array_dict

        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
