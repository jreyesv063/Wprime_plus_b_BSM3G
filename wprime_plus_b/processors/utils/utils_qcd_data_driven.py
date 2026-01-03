import numpy as np
import awkward as ak
from typing import Dict, Any
from coffea.analysis_tools import PackedSelection, Weights

# Utils
from wprime_plus_b.processors.utils.analysis_utils import delta_r_mask, fill_cutflow

# Corrections
from wprime_plus_b.corrections.tau import TauCorrector

# Top tagger
from wprime_plus_b.processors.utils.utils_topXfinder import get_topXfinder_masks

# Systematics: Object - level
from wprime_plus_b.processors.utils.utils_syst_var import Systematic_variation

# Selections
from wprime_plus_b.object_identification.tau_selection import select_good_taus
from wprime_plus_b.object_identification.bjet_selection import select_good_bjets
from wprime_plus_b.object_identification.wjet_selection import select_good_wjets
from wprime_plus_b.object_identification.fatjet_selection import select_good_fatjets
from wprime_plus_b.object_identification.lightjet_selection import select_good_lightjets
from wprime_plus_b.object_identification.met_selection import select_good_delta_phi_jet_met

# Histograms
from wprime_plus_b.processors.utils.histogram_utils import histograms_output_array

class QCD_data_driven:
    def __init__(
        self,
        self_main: Any,
        year: str = "2017",
        lepton_flavor: str = "tau",
        is_mc: bool = False,
        syst: str = "Nominal",
        criteria: dict = None,
        events: ak.Array = None,
        objects: list = None,
        object_variations: list = None, 
        weights_container: Weights = None,
        selections: PackedSelection = None,
        region_selection_list: list = None,
        output_metadata: Dict[str, Any] = None,
        delta_list: list = None,
        processor: str = "wplusjets"
    ):

        # Validate required inputs
        if is_mc and weights_container is None:
            raise ValueError("weights_container required for MC")
            
        # Store attributes
        self.year = year
        self.lepton_flavor = lepton_flavor
        self.processor = processor
        self.is_mc = is_mc
        self.syst = syst
        self.criteria = criteria
        self.events = events
        self.objects = objects
        self.object_variations = object_variations 
        self.jets = objects["jets"]
        self.weights_container = weights_container 
        self.selections = selections
        self.self_main = self_main

        self.region_selection_list = region_selection_list
        self.output_metadata = output_metadata
        self.delta_list = delta_list 

        # Define auxiliary CR names
        self.cr_names = ["cr_b", "cr_c", "cr_d"]


        # Containers filled after selection
        self.taus_per_cr = {}
        self.weights_containers = {}
        self.cuts = {}

        # Choose DeepTau ID suffix for fake rate SFs
        self.suffix = "2017v2p1" if year in ["2016APV", "2016", "2017", "2018"] else "2018v2p5"

        # Shorthand to access CR-dependent criteria
        self.criteria_X = self.criteria["data_driven_qcd_estimation"]


    def get_new_cutflow(self, region: str = "cr_b"):
        """
        Generate a new cutflow list for a specific control region.
        
        Parameters:
        -----------
        region : str
            Control region name (e.g., "cr_b", "cr_c", "cr_d")
            
        Returns:
        --------
        list
            Modified cutflow list with region-specific cuts
        """

        # Start with the cut list in the nominal control region
        cuts = self.region_selection_list.copy()


        # Define nominal cut names to look for
        #nominal_cut_names = ["one_tau", "bjet_veto", "one_bjet", "delta_phi_jet_met"]

        nominal_cut_names = ["one_tau", "delta_phi_jet_met"]

        if self.processor == "wplusjets":
            nominal_cut_names.append("bjet_veto") 
        
        elif self.processor in ["signal", "qcd_hadronic_closure"]:
            nominal_cut_names.append("one_bjet") 
        else:
            raise ValueError(
                f"Processor '{self.processor}' is not supported in this part of the analysis. "
                f"Expected values: 'qcd_hadronic_closure, wplusjets' or 'signal'."
            )


        # Inject CR-specific selections while preserving nominal ordering
        for name in nominal_cut_names:
            # Construct the CR-tagged cut name
            tag = f"{name}_{region}"

            # Find position of nominal cut if it exists
            if name in cuts:
                pos = cuts.index(name)
                # Replace nominal cut with region-specific cut
                cuts[pos] = tag

            else:
                # If nominal cut doesn't exist, append at the end
                cuts.append(tag)

        return cuts


    def get_region_selection_BCD(self, run_systematics: bool = False, is_mc: bool = True):
        """Apply CR-B/C/D selections and register masks in `PackedSelection`."""

        for cr in self.cr_names:
            # --- Tau object selection ---
            # The nominal cut is applied as baseline, with region-dependent
            # inversion of pass/fail conditions for fake-tau QCD CRs.
            good_taus_masks = select_good_taus(
                events = self.objects["events"],
                tau_pt_threshold = self.criteria["tau"][self.lepton_flavor]["pt"],
                tau_eta_threshold = self.criteria["tau"][self.lepton_flavor]["eta"],
                tau_dz_threshold = self.criteria["tau"][self.lepton_flavor]["dz"],
                tau_vs_jet_pass = self.criteria_X[cr][self.lepton_flavor]["fake_VSjet_pass"],
                tau_vs_jet_fail = self.criteria_X[cr][self.lepton_flavor]["fake_VSjet_fail"] if cr != "cr_b" else None,
                tau_vs_ele = self.criteria["tau"][self.lepton_flavor]["fake_VSe"],
                tau_vs_mu = self.criteria["tau"][self.lepton_flavor]["fake_VSmu"],
                prong = self.criteria["tau"][self.lepton_flavor]["prongs"],
                is_mc = self.is_mc,
            )
            good_taus = (
                (good_taus_masks["nominal"])
                & (delta_r_mask(self.objects["events"].Tau, self.objects["electrons"], threshold=self.criteria["cross_cleaning"][self.lepton_flavor]))
                & (delta_r_mask(self.objects["events"].Tau, self.objects["muons"], threshold=self.criteria["cross_cleaning"][self.lepton_flavor]))
            )           
            self.objects["taus"] =  self.objects["events"].Tau[good_taus]

            # --- Exactly-one-tau: mask ---
            self.selections.add(f"one_tau_{cr}", ak.num(self.objects["taus"]) == 1)

            # select good bjets
            good_bjets_masks = select_good_bjets(
                events=self.objects["events"],
                jets=self.objects["jets_veto"] ,
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
                & (delta_r_mask(self.objects["jets_veto"], self.objects["electrons"], threshold=self.criteria["cross_cleaning"][self.lepton_flavor]))
                & (delta_r_mask(self.objects["jets_veto"], self.objects["muons"], threshold=self.criteria["cross_cleaning"][self.lepton_flavor]))
                & (delta_r_mask(self.objects["jets_veto"], self.objects["taus"], threshold=self.criteria["cross_cleaning"][self.lepton_flavor]))
            )
            self.objects["bjets"] = self.objects["jets_veto"][good_bjets]

            # --- bjet: mask ---
            self.selections.add(f"one_bjet_{cr}", ak.num(self.objects["bjets"]) == 1)
            self.selections.add(f"bjet_veto_{cr}", ak.num(self.objects["bjets"]) == 0)

            
            # select good jets
            good_jets_masks = select_good_lightjets(
                events=self.objects["events"],
                jets=self.objects["jets_veto"],
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
                & (delta_r_mask(self.objects["jets_veto"], self.objects["electrons"], threshold=self.criteria["cross_cleaning"][self.lepton_flavor]))
                & (delta_r_mask(self.objects["jets_veto"], self.objects["muons"], threshold=self.criteria["cross_cleaning"][self.lepton_flavor]))
                & (delta_r_mask(self.objects["jets_veto"], self.objects["taus"], threshold=self.criteria["cross_cleaning"][self.lepton_flavor]))
            )
            self.objects["jets"] = self.objects["jets_veto"][good_jets]


            # select good tops
            good_fatjets_masks = select_good_fatjets(
                fatjets = self.objects["events"].FatJet,
                year = self.year,
                fatjet_pt_threshold = self.criteria["fatjet"][self.lepton_flavor]["pt"],
                fatjet_eta_threshold = self.criteria["fatjet"][self.lepton_flavor]["eta"],
                TvsQCD = self.criteria["fatjet"][self.lepton_flavor]["particleNet_Top_Nominal"],
                is_mc=self.is_mc,
            )
            good_fatjets = (
                good_fatjets_masks["nominal"]
                & (delta_r_mask(self.objects["events"].FatJet, self.objects["electrons"], threshold = 2*self.criteria["cross_cleaning"][self.lepton_flavor]))
                & (delta_r_mask(self.objects["events"].FatJet, self.objects["muons"], threshold = 2*self.criteria["cross_cleaning"][self.lepton_flavor]))
                & (delta_r_mask(self.objects["events"].FatJet, self.objects["taus"], threshold = 2*self.criteria["cross_cleaning"][self.lepton_flavor]))
                & (delta_r_mask(self.objects["events"].FatJet, self.objects["bjets"], threshold = 2*self.criteria["cross_cleaning"][self.lepton_flavor]))
                & (delta_r_mask(self.objects["events"].FatJet, self.objects["jets"], threshold = 2*self.criteria["cross_cleaning"][self.lepton_flavor]))
            )   
            self.objects["fatjets"]  = self.objects["events"].FatJet[good_fatjets]


            # select good W jets
            good_wjets_masks = select_good_wjets(
                wjets = self.objects["events"].FatJet,
                year = self.year,
                w_pt_threshold = self.criteria["wjet"][self.lepton_flavor]["pt"],
                w_eta_threshold = self.criteria["wjet"][self.lepton_flavor]["eta"],
                WvsQCD = self.criteria["wjet"][self.lepton_flavor]["particleNet_W_Nominal"],
                is_mc=self.is_mc,
            )
            good_wjets = (
                good_wjets_masks["nominal"]
                & (delta_r_mask(self.objects["events"].FatJet, self.objects["electrons"], threshold = 2*self.criteria["cross_cleaning"][self.lepton_flavor]))
                & (delta_r_mask(self.objects["events"].FatJet, self.objects["muons"], threshold = 2*self.criteria["cross_cleaning"][self.lepton_flavor]))
                & (delta_r_mask(self.objects["events"].FatJet, self.objects["taus"], threshold = 2*self.criteria["cross_cleaning"][self.lepton_flavor]))
                & (delta_r_mask(self.objects["events"].FatJet, self.objects["bjets"], threshold = 2*self.criteria["cross_cleaning"][self.lepton_flavor]))
                & (delta_r_mask(self.objects["events"].FatJet, self.objects["jets"], threshold = 2*self.criteria["cross_cleaning"][self.lepton_flavor]))
                & (delta_r_mask(self.objects["events"].FatJet, self.objects["fatjets"], threshold = 2*self.criteria["cross_cleaning"][self.lepton_flavor]))
            )   
            self.objects["wjets"] = self.objects["events"].FatJet[good_wjets]

            # --- MC weights + tau ID SF corrections (MC only) ---
            if self.is_mc:
                """
                It is only necessary to recalculate the weights of the taus, as they are the only objects that can change in CRB, CRC, and CRD.
                """
                # New weights container
                weights_cr = Weights(len(self.events), storeIndividual=True)

                # Make a copy of the weights, except for those of the tau ID
                weights_cr.add("baseline_weights", 
                    self.weights_container.partial_weight(
                        exclude=[f"CMS_fake_t_DeepTau{self.suffix}_VSjet"]
                    )
                )

                # Add new tau correction
                tau_corrector = TauCorrector(
                    taus = self.objects["taus"],
                    weights = weights_cr,
                    year = self.year,
                    tau_vs_jet = self.criteria_X[cr][self.lepton_flavor]["fake_VSjet_pass"],
                    tau_vs_ele = self.criteria["tau"][self.lepton_flavor]["fake_VSe"],
                    tau_vs_mu = self.criteria["tau"][self.lepton_flavor]["fake_VSmu"],
                    variation = self.syst,
                )
                tau_corrector.add_id_weight_DeepTau2017v2p1VSjet()

                self.weights_containers[cr] = weights_cr

            else:
                self.weights_containers[cr] = self.weights_container

            # --- Δφ(jet, MET) event mask ---
            # The nominal Δφ cut is used as baseline, with optional inversion
            # depending on the auxiliary control region definition.
            delta_phi_mask = select_good_delta_phi_jet_met(
                events = self.events,
                jets = self.objects["jets"],
                delta_phi_cut = self.criteria["met"][self.lepton_flavor]["delta_phi_jets_met"],
                invert_delta_phi_cut= self.criteria_X[cr][self.lepton_flavor]["invert_delta_phi"]
            )

            # Save selection criteria
            self.selections.add(f"delta_phi_jet_met_{cr}", delta_phi_mask)

            self.cuts[cr] = self.get_new_cutflow(cr)

        for cr in self.cuts.keys():
            cuts_name_X = self.cuts[cr]
            weights_X =  self.weights_containers[cr]

            self.output_metadata.update({f"sumw_{cr}": ak.sum(self.weights_containers[cr].weight())})


            self.output_metadata.update({f"cutflow_{cr}": {}})
            self.output_metadata.update({f"cutflow_{cr}_raw": {}})

            fill_cutflow(metadata = self.output_metadata, cut_name = "sumw", table_name = f"cutflow_{cr}", weights = weights_X.weight())

            # Selecciones temporales
            selections_temp = PackedSelection(dtype='uint64') # New


            selections = []        
            for cut_name_X in cuts_name_X:
                selections.append(cut_name_X)
                current_selection = self.selections.all(*selections)
                selections_temp.add(cut_name_X, current_selection) # New
                fill_cutflow(metadata = self.output_metadata, cut_name = cut_name_X, table_name = f"cutflow_{cr}", weights = weights_X.weight()[current_selection])


            # Final mask
            #region_selection_mask = self.selections.all(*cuts_name_X)
            region_selection_mask = selections_temp.all(*cuts_name_X)
            nevents_after = ak.sum(region_selection_mask)
            region_selection_weights = weights_X.weight()[region_selection_mask]

            if nevents_after == 0:
                fill_cutflow(metadata = self.output_metadata, cut_name = f"fail_top_tagger_{cr}", table_name = f"cutflow_{cr}", weights = region_selection_weights)

                # save weighted events to metadata
                self.output_metadata.update({
                    f"weighted_final_nevents_{cr}": ak.sum(region_selection_weights),
                    f"raw_final_nevents_{cr}": nevents_after,
                })

            else:
                # =============================================================
                #                   Top tagger mask
                # =============================================================
                mask_top, masks, njets_no_top, tops, selected_objects = get_topXfinder_masks(
                    lepton_flavor = self.lepton_flavor,
                    region_mask = region_selection_mask,
                    objects = self.objects,
                    top_tagger_cases = self.criteria["top_tagger"][self.lepton_flavor]["cases"],
                    cross_cleaning = self.criteria["cross_cleaning"][self.lepton_flavor],
                    invert_topXfinder = self.criteria["top_tagger"][self.lepton_flavor]["invert_top_tagger"]
                )

                weights = region_selection_weights[mask_top]

                fill_cutflow(metadata = self.output_metadata, cut_name = f"fail_top_tagger_{cr}", table_name = f"cutflow_{cr}", weights = weights)

                nevents_top_tagger = ak.sum(mask_top)


                # save weighted events to metadata
                self.output_metadata.update({
                    f"weighted_final_nevents_{cr}": ak.sum(region_selection_weights[mask_top]),
                    f"raw_final_nevents_{cr}": nevents_top_tagger,
                })

                if nevents_top_tagger > 0:
                    histograms_output_array(
                        self_main = self.self_main,
                        lepton_flavor =  self.lepton_flavor,
                        njets_no_top = njets_no_top,
                        tops = tops,
                        objects = selected_objects,
                        mask = mask_top,
                        name = cr,
                    )

                    self.self_main.add_feature(f"weights_{cr}", region_selection_weights[mask_top])

            # ============================================================== 
            #                     Systematics variations
            # ============================================================== 

            if run_systematics and is_mc:
                syst_var = Systematic_variation(
                    lepton_flavor = self.lepton_flavor, 
                    cut_names = self.cuts[cr],  
                    criteria = self.criteria,
                    selections = selections_temp,   
                    weights_container = self.weights_container,                            
                    metadata = self.output_metadata, 
                    objects = self.objects,
                    object_variations = self.object_variations, 
                    delta_list = self.delta_list,
                    processor = self.processor
                )

                syst_var.get_syst_variation_mask(self_main = self.self_main, table_name = f"cutflow_{cr}", nworkers =  self.criteria["top_tagger"][self.lepton_flavor]["nworkers"], name = cr)                  

                if nevents_top_tagger > 0:
                    syst_var.get_syst_variation_event_level(name = cr, region_mask = region_selection_mask, mask = mask_top, self_main = self.self_main)