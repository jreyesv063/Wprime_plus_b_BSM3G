import gc  
import numpy as np
import awkward as ak
from typing import Dict, Any
from coffea.analysis_tools import PackedSelection, Weights


# =======================================
# Corrections: event-level
# =======================================
from wprime_plus_b.corrections.pileup import add_pileup_weight
from wprime_plus_b.corrections.pujetid import add_pujetid_weight
from wprime_plus_b.corrections.met import add_met_trigger_corrections
from wprime_plus_b.corrections.l1prefiring import add_l1prefiring_weight
from wprime_plus_b.corrections.top_boost import add_top_boost_corrections
from wprime_plus_b.corrections.top_pt_reweighting import add_TopPtReweighting
from wprime_plus_b.corrections.wjets_topjets import add_QCD_vs_W_weight, add_QCD_vs_Top_weight


from wprime_plus_b.corrections.tau import TauCorrector
from wprime_plus_b.corrections.btag import BTagCorrector


# =======================================
# Selections: objects
# =======================================
from wprime_plus_b.object_identification.tau_selection import select_good_taus
from wprime_plus_b.object_identification.met_selection import select_good_delta_phi_jet_met


# =======================================
# General cuts
# =======================================
# Top tagger
from wprime_plus_b.general_selections.triggers import get_trigger_mask
from wprime_plus_b.object_identification.top_selection import select_top_tagger


# =======================================
# Systematic variations
# =======================================
from wprime_plus_b.processors.utils.utils_syst_var import Systematics

# =======================================
#  Plots
# =======================================
from wprime_plus_b.processors.utils.histograms import Histograms

# =======================================
# Utils
# =======================================
from wprime_plus_b.processors.utils.analysis_utils import cross_cleaning, fill_sumw, fill_cutflow, get_mask_until_object 

class QCD_data_driven:
    def __init__(
        self,
        cc=0.4,
        dataset="", 
        year="2017", 
        objects=None, 
        criteria=None, 
        selections=None, 
        variations=None, 
        processor="test",
        output_hist=None, 
        lepton_flavor="tau",
        output_metadata=None, 
        weights_container=None, 
        region_selection_cuts=None, 
        systematic_variations=False
    ):
        self.cc = cc
        self.year = year
        self.dataset = dataset
        self.objects = objects  
        self.criteria = criteria
        self.processor = processor
        self.variations = variations
        self.selections = selections
        self.output_hist = output_hist
        self.syst = systematic_variations
        self.lepton_flavor = lepton_flavor
        self.output_metadata = output_metadata
        self.weights_container = weights_container
        self.is_mc = hasattr(objects["events"], "genWeight") 
        
        self.region_selection_cuts = region_selection_cuts
        self.suffix = "2017v2p1" if year in ["2016APV", "2016", "2017", "2018"] else "2018v2p5"

        self.cuts_map = {}
        self.selections_BCD = PackedSelection(dtype='uint64')

    def get_new_cutflow(self):
        object_cut_names = {"electron", "muon", "tau", "bjet", "lightjet", "jet", "top_tagger"}
        
        for reg in ["cr_b", "cr_c", "cr_d"]:
            self.cuts_map[reg] = [
                f"{cut}_{reg}" if any(obj in cut for obj in object_cut_names) else cut
                for cut in self.region_selection_cuts
            ]
            # Añadir máscaras existentes de forma eficiente
            for name in self.cuts_map[reg]:
                if name in self.selections.names:
                    self.selections_BCD.add(name, self.selections.all(name))

    def get_object_mask(self):
        
        for cr in ["cr_b", "cr_c", "cr_d"]:
            """
            We create a new dictionary with references to the same awkward arrays.
            Awkward is immutable, so we don't need deep copying to protect the data.
            """
            objects_region = {k: v for k, v in self.objects.items()}
            
            # Only copy variations if they exist
            objects_var = self.variations.copy() if self.variations is not None else None
            
            weights_cr = Weights(len(objects_region["events"]), storeIndividual=True)
            
            # Obtain definition of regions B, C, and D based on the criteria defined in the config file
            cr_criteria = self.criteria["data_driven_qcd_estimation"][cr][self.lepton_flavor]
            
            if cr_criteria["fake_VSjet_fail"] is not None or cr_criteria["fake_VSjet_pass"] != self.criteria["tau"][self.lepton_flavor]["fake_VSjet_pass"]: 
                good_taus_masks = select_good_taus(
                    events=objects_region["events"],
                    tau_pt_threshold=self.criteria["tau"][self.lepton_flavor]["pt"],
                    tau_eta_threshold=self.criteria["tau"][self.lepton_flavor]["eta"],
                    tau_dz_threshold=self.criteria["tau"][self.lepton_flavor]["dz"],
                    tau_vs_jet_pass=cr_criteria["fake_VSjet_pass"],
                    tau_vs_jet_fail=cr_criteria["fake_VSjet_fail"],
                    tau_vs_ele=self.criteria["tau"][self.lepton_flavor]["fake_VSe"],
                    tau_vs_mu=self.criteria["tau"][self.lepton_flavor]["fake_VSmu"],
                    prong=self.criteria["tau"][self.lepton_flavor]["prongs"],
                    syst_var=self.syst
                )
                
                objects_region["taus"] = objects_region["events"].Tau[good_taus_masks["nominal"]] 
                
                if objects_var is not None:
                    # None is in data samples
                    # Save new tau mask in variations for systematic evaluation
                    objects_var["tau"] = good_taus_masks

            
            objects_cleaned = cross_cleaning(objects_region, self.cc)

            # Delta Phi criteria for CRs BCD
            delta_phi_masks = select_good_delta_phi_jet_met(
                events=objects_cleaned["events"], 
                jets=objects_cleaned["lightjets"], 
                delta_phi_cut=self.criteria["met"][self.lepton_flavor]["delta_phi_jets_met"],
                invert_delta_phi_cut=cr_criteria["invert_delta_phi"],
                syst_var=self.syst
            )
           
            
            if objects_var is not None:
                # None is in data samples
                # Save new delta phi mask in variations for systematic evaluation
                objects_var["delta_phi_jet_met"] = delta_phi_masks

            # Add selections for CRs
            self.selections_BCD.add(f"one_tau_{cr}", ak.num(objects_cleaned["taus"]) == 1)
            self.selections_BCD.add(f"delta_phi_jet_met_{cr}", delta_phi_masks["nominal"])
            self.selections_BCD.add(f"muon_veto_{cr}", ak.num(objects_cleaned["muons"]) == 0)
            self.selections_BCD.add(f"one_muon_{cr}", ak.num(objects_cleaned["muons"]) == 1)
            self.selections_BCD.add(f"electron_veto_{cr}", ak.num(objects_cleaned["electrons"]) == 0)   
            self.selections_BCD.add(f"bjet_veto_{cr}", ak.num(objects_cleaned["bjets"]) == 0)
            self.selections_BCD.add(f"one_bjet_{cr}", ak.num(objects_cleaned["bjets"]) == 1)          
       
                

            if cr_criteria["fake_VSjet_fail"] is not None or cr_criteria["fake_VSjet_pass"] != self.criteria["tau"][self.lepton_flavor]["fake_VSjet_pass"]:
                # Recalculate top tagger mask since the input collection of taus has changed, which can affect the cross-cleaning and thus the top tagger candidates.
                objects_cleaned, top_tagger_mask = select_top_tagger(
                    objects=objects_cleaned,
                    region_mask=ak.ones_like(objects_cleaned["events"].run, dtype=bool), # All events to be evaluated
                    cross_cleaning=self.cc,
                    criteria=self.criteria["top_tagger"][self.lepton_flavor]
                )
            
                self.selections_BCD.add(f"top_tagger_{cr}", top_tagger_mask)

            else:
                # Use the existing top tagger mask since the tau selection didn't change in a way that would affect it. This avoids unnecessary recalculation.
                self.selections_BCD.add(f"top_tagger_{cr}", self.selections.all("top_tagger"))
            

            if self.lepton_flavor == "tau" and self.is_mc:    
                has_bjet = any("bjet" in cut for cut in self.cuts_map[cr])
                has_top  = any("top_tagger" in cut for cut in self.cuts_map[cr])


                if cr_criteria["fake_VSjet_fail"] is not None or cr_criteria["fake_VSjet_pass"] != self.criteria["tau"][self.lepton_flavor]["fake_VSjet_pass"]:
                    # Since the subset of taus changes, it is necessary to recalculate the identification weights.
                    exclude_list = [
                        f"CMS_fake_t_DeepTau{self.suffix}_VSjet_{self.year}", f"CMS_fake_t_DeepTau{self.suffix}_VSmu_{self.year}", f"CMS_fake_t_DeepTau{self.suffix}_VSe_{self.year}",
                        f"CMS_eff_j_ParticleNet_Top_Nominal_{self.year}", f"CMS_eff_j_ParticleNet_W_Nominal_{self.year}",
                        f"CMS_btag_heavy_{self.year}", f"CMS_btag_light_{self.year}",                         
                        f"top_boost_weight_{self.lepton_flavor}_{self.year}",
                        f"CMS_eff_j_PUJetID_eff_{self.year}",
                        f"CMS_eff_MET_trigger_{self.year}",
                        f"top_pt_reweighting_{self.year}",
                        f"CMS_pileup_{self.year}",
                        *(
                            [f"CMS_l1_prefiring_{self.year}"] if self.year in ["2016APV", "2016", "2017"] else []
                        )
                    ]

                    for name, w in self.weights_container._weights.items():
                        if name in exclude_list:
                            continue

                        up = self.weights_container._modifiers.get(name + "Up")
                        down = self.weights_container._modifiers.get(name + "Down")

                        
                        if up is not None and down is not None:
                            weights_cr.add(
                                name,
                                weight=w,
                                weightUp=up,
                                weightDown=down,
                            )
                        else:
                            weights_cr.add(name, weight=w)    

                    # ************************************
                    #    Pileup, L1prefiring,
                    #    top pt reweighting
                    #    MET trigger
                    # ************************************
                    add_pileup_weight(objects_cleaned["events"], weights_cr, self.year)
                    add_l1prefiring_weight(objects_cleaned["events"], weights_cr, self.year)
                    add_TopPtReweighting(objects_cleaned["events"], weights_cr, self.dataset, self.year)   

                    trigger_names, trigger_mask = get_trigger_mask(
                        events=objects_cleaned['events'],
                        lepton_flavor = 'tau',
                        year = self.year,
                        trigger_case = self.criteria["trigger"]['tau']["main"],
                        Or_HLT = self.criteria["trigger"]['tau']["OR_trigger"]
                    )
                    
                    add_met_trigger_corrections(
                        trigger_mask, 
                        self.dataset, 
                        objects_cleaned["met"], 
                        weights_cr, 
                        self.year,
                        trigger_mask = get_mask_until_object(selections=self.selections_BCD, cuts=self.cuts_map[cr], obj_name="trigger", include_cut=False, only_cut=True)
                    )   


                    # ************************************
                    #          Pileup Jet ID
                    # ************************************
                    add_pujetid_weight(
                        jets=objects_cleaned["jets"],
                        weights=weights_cr,
                        year=self.year,
                        working_point = self.criteria["jet"][self.lepton_flavor]["pileup_id"],
                        jet_mask = get_mask_until_object(selections=self.selections_BCD, cuts= self.cuts_map[cr], obj_name="top_tagger" if has_top and not has_bjet else "bjet", include_cut=False, only_cut=True)
                    )

                    # ************************************
                    #              Bjets
                    # ************************************
                    btag_corrector = BTagCorrector(
                        jets=objects_cleaned["jets"],
                        dataset = self.dataset,
                        year=self.year,
                        tagger="deepJet",                
                        weights=weights_cr,
                        pass_working_point=self.criteria["bjet"][self.lepton_flavor]["btag_wp_pass"],
                        fail_working_point=self.criteria["bjet"][self.lepton_flavor]["btag_wp_pass"] if self.processor=="qcd_hadronic_closure" else None,
                        bjet_mask = get_mask_until_object(selections=self.selections_BCD, cuts=self.cuts_map[cr], obj_name="top_tagger" if has_top and not has_bjet else "bjet", include_cut=False, only_cut=True),
                        mask_eff_btag = get_mask_until_object(selections=self.selections_BCD, cuts=self.cuts_map[cr], obj_name="top_tagger" if has_top and not has_bjet else "bjet", include_cut=False, only_cut=False)
                    )            
                    # add b-tagging weights
                    btag_corrector.add_btag_weights(flavor="bc", correlated=False)
                    btag_corrector.add_btag_weights(flavor="light", correlated=False)
                
                    # ************************************
                    # Hadronic taus
                    # ************************************
                    tau_corrector = TauCorrector(
                        taus=objects_cleaned["taus"],
                        weights=weights_cr,
                        year=self.year,
                        tau_vs_jet=cr_criteria["fake_VSjet_pass"],
                        tau_vs_ele=self.criteria["tau"][self.lepton_flavor]["fake_VSe"],
                        tau_vs_mu=self.criteria["tau"][self.lepton_flavor]["fake_VSmu"],
                        tau_mask = get_mask_until_object(selections=self.selections_BCD, cuts=self.cuts_map[cr], obj_name="tau", include_cut=False, only_cut=True)
                    )
                    tau_corrector.add_id_weight_DeepTau2017v2p1VSe()
                    tau_corrector.add_id_weight_DeepTau2017v2p1VSmu()
                    tau_corrector.add_id_weight_DeepTau2017v2p1VSjet()

                    # ************************************
                    # Top and W algorithm identification
                    # ************************************
                    # Top candidates participate in cases = 5, 6, 7, 8, 13
                    add_QCD_vs_Top_weight(
                        topjets=objects_cleaned["topjets"],
                        weights=weights_cr,
                        year=self.year,
                        working_point_topjet = self.criteria["topjet"][self.lepton_flavor]["particleNet_Top_Nominal"],
                        top_mask = (
                            (objects_cleaned["events"].top_tagger_case_id == 5) 
                            | (objects_cleaned["events"].top_tagger_case_id == 6) 
                            | (objects_cleaned["events"].top_tagger_case_id == 7) 
                            | (objects_cleaned["events"].top_tagger_case_id == 8) 
                            | (objects_cleaned["events"].top_tagger_case_id == 13)
                        )
                    )

                    # W candidates participate in cases = 3, 4, 11, 12
                    add_QCD_vs_W_weight(
                        wjets=objects_cleaned["wjets"],
                        weights=weights_cr,
                        year=self.year,
                        working_point_wjet = self.criteria["wjet"][self.lepton_flavor]["particleNet_W_Nominal"],
                        W_mask = (
                            (objects_cleaned["events"].top_tagger_case_id == 3) 
                            | (objects_cleaned["events"].top_tagger_case_id == 4) 
                            | (objects_cleaned["events"].top_tagger_case_id == 11) 
                            | (objects_cleaned["events"].top_tagger_case_id == 12)
                        )
                    )

                    add_top_boost_corrections(
                        objects=objects_cleaned, 
                        lepton_flavor=self.lepton_flavor, 
                        dataset=self.dataset, 
                        weights=weights_cr, 
                        year=self.year
                    ) 
                
            else:
                # Use the weights of the nominal selection since the tau selection didn't change in a way that would affect the weights. This avoids unnecessary recalculation.
                weights_cr = self.weights_container

            # Process histograms IMMEDIATELY to avoid saving ‘objects_cleaned’ in a map
            self._fill_results(cr, objects_cleaned, weights_cr, objects_var)
            
            # Explicit cleanup for each iteration of the CR
            del objects_region, objects_cleaned, weights_cr
            gc.collect()

    def _fill_results(self, cr, objects, weights, objects_var):
        """Auxiliary method for processing data for each CR and quickly freeing up memory"""
        self.output_metadata[cr] = {}
        
        #self.output_metadata[cr].update({"sumw": ak.sum(weights.partial_weight(include=["genweight"]))})
        fill_sumw(weights, self.is_mc, self.output_metadata[cr])

        self.output_metadata[cr].update({
            "weight_statistics": {k: v for k, v in weights.weightStatistics.items()}
        })
        
        fill_cutflow(self.cuts_map[cr], self.selections_BCD, f"cutflow_{cr}", 
                     self.output_metadata[cr], weights.weight())

        self.output_hist[cr] = {}
        hist = Histograms(self.lepton_flavor, self.processor, objects, weights.weight(), self.selections_BCD, self.cuts_map[cr])
        
        self.output_hist[cr]["nominal"] = hist.fill_histograms()
        self.output_hist[cr]["nominal"]["count"] = 1
        self.output_hist[cr]["nominal"]["sumw_all_weights"] = np.sum(weights.partial_weight(include=["genweight"]))

        # Process systematically right here if necessary to avoid keeping large objects in memory for too long
        if self.syst and self.is_mc:
            systematic_variations = Systematics(
                cr=cr, 
                cc = self.cc, 
                year=self.year, 
                weights=weights, 
                objects=objects, 
                criteria=self.criteria, 
                variations=objects_var, 
                processor=self.processor, 
                table_name=f"cutflow_{cr}",
                selections=self.selections_BCD, 
                histograms=self.output_hist[cr],
                metadata=self.output_metadata[cr],
                lepton_flavor=self.lepton_flavor, 
                cut_names=self.cuts_map[cr]
            )
            systematic_variations.object_level()
            systematic_variations.event_level()

    def get_BCD_controlRegions(self):
        self.get_new_cutflow()
        self.get_object_mask()
        