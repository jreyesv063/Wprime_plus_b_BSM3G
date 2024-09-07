import numpy as np
import awkward as ak
import json
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from wprime_plus_b.processors.utils.analysis_utils import chi2_test, delta_r_mask, pdg_masses, tagger_constants
from wprime_plus_b.selections.top_tagger.cases_top_tagger_config import top_tagger_mW_mTop_Njets_selection


# --------------------------
# Top tagger
# --------------------------

class topXfinder:
    def __init__(
        self,
        lepton_flavor,
        bjets,
        jets,
        fatjets,
        wjets,
 #       year: str = "2017",
 #       year_mod: str = "",
        cc: float = 0.4,
    ) -> None:
        
        
        # year
#        self.year = year
#        self.year_mod = year_mod        
        
        self.cross_cleaning = cc
        
        self.bjets = bjets
        self.jets = jets
        self.fatjets = fatjets
        self.wjets = wjets

        self.lepton_flavor = lepton_flavor

        self.top_mass_pdg, self.w_mass_pdg = pdg_masses()
    
        
    """
    https://gitlab.cern.ch/topsXFinder/topsXFinder/-/tree/master
    
    1. Fully boosted top

    2. Partially boosted top: boosted W + b jet

    3. Resolved hadronic top: 2 AK4 jet +b jet

    4. Resolved leptonic top: 1 lepton(e/mu) + MET + b jet

    """
    ######################################
    #########  1 Jet #####################
    ######################################
    # Case 1
    def Scenario_1jet_unresolve(self):
        
        top_sigma, top_low_mass, top_up_mass, w_sigma, w_low_mass, w_up_mass, chi2 = tagger_constants("boosted")
        
        
        # We select scenarios with 1 fatjet
        initial_mask = (
            (ak.num(self.bjets) == 0)
            & (ak.num(self.jets) == 0)
            & (ak.num(self.fatjets) == 1)
            & (ak.num(self.wjets) == 0)
        )
        
        mask_objects = ak.fill_none(initial_mask, False)
        
        fatjets_masked = self.fatjets.mask[mask_objects]
    
        # ------
        # Topjet identification
        # ------
        good_top_mass = (
            (fatjets_masked.mass > top_low_mass) 
            & (fatjets_masked.mass < top_up_mass) 
        )
        
        tops = fatjets_masked.mask[good_top_mass]
        
        mask_final = ak.fill_none(ak.firsts(good_top_mass), False)
        top = ak.fill_none(ak.firsts(tops.mass), 0)
        

        return top, mask_final
    
    ######################################
    #########  2 Jet #####################
    ######################################
        
    # Case 2   
    def Scenario_2jets_unresolve(self):
       
        top_sigma, top_low_mass, top_up_mass, w_sigma, w_low_mass, w_up_mass, chi2 = tagger_constants("boosted")

        # We select scenarios with 1b and 1 fatjet
        initial_mask = (
            (ak.num(self.bjets) == 1)
            & (ak.num(self.jets) == 0)
            & (ak.num(self.fatjets) == 1)
            & (ak.num(self.wjets) == 0)
        )
        
        mask_objects = ak.fill_none(initial_mask, False)
        
        jet_bs = self.bjets.mask[mask_objects]
        jet_fatjets = self.fatjets.mask[mask_objects]
        
        
        # -------------------------------------------
        # Cross cleaning
        cross_cleaning = (
            (jet_bs.delta_r(jet_fatjets) > 2*self.cross_cleaning)
        )

        b1_cc = jet_bs.mask[cross_cleaning]            
        fatjet_cc = jet_fatjets.mask[cross_cleaning]            

        
        # ------
        # Topjet identification
        # ------
        good_top_mass = (
            (fatjet_cc.mass > top_low_mass) 
            & (fatjet_cc.mass < top_up_mass) 
        )
        
        
        tops = fatjet_cc.mask[good_top_mass]
        mask_final = ak.fill_none(ak.firsts(good_top_mass), False)
        top = ak.fill_none(ak.firsts(tops.mass), 0)
        
 
        return top, mask_final
    
    
    # ----------------------------------------
    #  Scenario V (W = 1  +  b = 1)
    # ----------------------------------------
    # Case 3
    def Scenario_2jets_partiallyresolve(self):
        
        top_sigma, top_low_mass, top_up_mass, w_sigma, w_low_mass, w_up_mass, chi2 = tagger_constants("Partially_boosted")
        
        # We select scenarios with 1b and 1 wjet
        initial_mask = (
            (ak.num(self.bjets) == 1)
            & (ak.num(self.jets) == 0)
            & (ak.num(self.fatjets) == 0)
            & (ak.num(self.wjets) == 1)
        )
        
        mask_objects = ak.fill_none(initial_mask, False)
        
        jet_bs = self.bjets.mask[mask_objects]
        jet_wjets = self.wjets.mask[mask_objects]
        
        jet_b1 = ak.firsts(jet_bs)
        jet_w1 = ak.firsts(jet_wjets)
        
        # -------------------------------------------
        # Cross cleaning
        cross_cleaning = (
            (jet_b1.delta_r(jet_w1) > 2*self.cross_cleaning)           
        )

        b1_cc = jet_b1.mask[cross_cleaning]            
        w1_cc = jet_w1.mask[cross_cleaning]            

        
        # ------
        # Wjet identification
        # ------
        good_w_mass = (
            (w1_cc.mass > w_low_mass) 
            & (w1_cc.mass < w_up_mass) 
        )
        
        w = w1_cc.mask[good_w_mass]
        b1 = b1_cc.mask[good_w_mass]

        
        # ---------------------------------------
        # We will define the top using the b + w
        # ---------------------------------------
        dijet= w + b1
        
        good_tops = (
                (dijet.mass > top_low_mass)
                & (dijet.mass <  top_up_mass)
        )   
        
        filtered_tops = dijet.mask[good_tops]
        filtered_ws = w.mask[good_tops]
        
        # ---------------------------------------
        # chi2 criteria
        # ---------------------------------------
        chi2_cal = chi2_test(filtered_tops, filtered_ws, top_sigma, w_sigma, self.top_mass_pdg, self.w_mass_pdg)
    
        
        
        good_chi2 = (chi2_cal < chi2)
        tops = filtered_tops.mask[good_chi2]
        
        mask_final = ak.fill_none(good_chi2, False)
        top = ak.fill_none(tops.mass, 0)
        
        
        return top, mask_final
    
    
    
    ######################################
    #########  3 Jet #####################
    ######################################
        
    # ----------------------------------------
    #  Scenario V (W = 1  +  b = 2)
    # ----------------------------------------
    # Case 4
    def Scenario_3jets_partiallyresolve(self):
        
        top_sigma, top_low_mass, top_up_mass, w_sigma, w_low_mass, w_up_mass, chi2 = tagger_constants("Partially_boosted")
        
        # We select scenarios with 2b and 1 wjet
        initial_mask = (
            (ak.num(self.bjets) == 2)
            & (ak.num(self.jets) == 0)
            & (ak.num(self.fatjets) == 0)
            & (ak.num(self.wjets) == 1)
        )
        
        mask_objects = ak.fill_none(initial_mask, False)
        
        jet_bs = self.bjets.mask[mask_objects]
        jet_wjets = self.wjets.mask[mask_objects]
        
        jet_b1 = ak.firsts(jet_bs)
        jet_b2 = ak.pad_none(jet_bs, 2)[:, 1]


        jet_w1 = ak.firsts(jet_wjets)
        
        # -------------------------------------------
        # Cross cleaning
        cross_cleaning = (
            (jet_b1.delta_r(jet_b2) > self.cross_cleaning)
            & (jet_b1.delta_r(jet_w1) > 2*self.cross_cleaning)
            & (jet_b2.delta_r(jet_w1) > 2*self.cross_cleaning)
        )

        b1_cc = jet_b1.mask[cross_cleaning]            
        b2_cc = jet_b2.mask[cross_cleaning]      

        w1_cc = jet_w1.mask[cross_cleaning]            

        
        # ------
        # Wjet identification
        # ------
        good_w_mass = (
            (w1_cc.mass > w_low_mass) 
            & (w1_cc.mass < w_up_mass) 
        )
        
        w = w1_cc.mask[good_w_mass]
        b1 = b1_cc.mask[good_w_mass]
        b2 = b2_cc.mask[good_w_mass]
        
        # ---------------------------------------
        # We will define the top using the b + w
        # ---------------------------------------
        condition = (
            ((w + b1).mass > top_low_mass) 
            & ((w + b1).mass < top_up_mass)
        )
        
        dijet = ak.where(condition,
                       w + b1,
                       w + b2
        )

        good_tops = (
                (dijet.mass > top_low_mass)
                & (dijet.mass <  top_up_mass)
        )   
        
        filtered_tops = dijet.mask[good_tops]
        filtered_ws = w.mask[good_tops]
        
        
        # ---------------------------------------
        # chi2 criteria
        # ---------------------------------------
        chi2_cal = chi2_test(filtered_tops, filtered_ws,  top_sigma, w_sigma, self.top_mass_pdg, self.w_mass_pdg)

        
        good_chi2 = (chi2_cal < chi2)
        tops = filtered_tops.mask[good_chi2]
        
        mask_final = ak.fill_none(good_chi2, False)
        top = ak.fill_none(tops.mass, 0)
        
        return top, mask_final
    
    

    # Case 5
    def Scenario_3jets_resolve(self):
        
        top_sigma, top_low_mass, top_up_mass, w_sigma, w_low_mass, w_up_mass, chi2 = tagger_constants("hadronic")
        
        # We select scenarios with 1b and 2jets
        initial_mask = ( 
            (ak.num(self.bjets) == 1)
            & (ak.num(self.jets) == 2)
            & (ak.num(self.fatjets) == 0)
            & (ak.num(self.wjets) == 0)
        )     
        
        mask_objects = ak.fill_none(initial_mask, False)

        
        
        jet_bs = self.bjets.mask[mask_objects]
        jet_jets = self.jets.mask[mask_objects]
        

        jet_b1 = ak.firsts(jet_bs)


        jet_l1 = ak.firsts(jet_jets)
        jet_l2 = ak.pad_none(jet_jets, 2)[:, 1]


        # -------------------------------------------
        # Cross cleaning
        cross_cleaning = (
            (jet_l1.delta_r(jet_l2) > self.cross_cleaning)
            & (jet_l1.delta_r(jet_b1) > self.cross_cleaning)
            & (jet_l2.delta_r(jet_b1) > self.cross_cleaning)
        )

        b1_cc = jet_b1.mask[cross_cleaning]            
       
        l1_cc = jet_l1.mask[cross_cleaning]            
        l2_cc = jet_l2.mask[cross_cleaning]   


        # ---------------------------------------
        # We will define the W using 2j
        # ---------------------------------------

        dijet_cc = l1_cc + l2_cc

        good_w_mass = (
            (dijet_cc.mass > w_low_mass) 
            & (dijet_cc.mass < w_up_mass) 
        )       

        w = dijet_cc.mask[good_w_mass] 
        b1 = b1_cc.mask[good_w_mass]


        # ---------------------------------------
        # We will define the top using the b + 2j
        # ---------------------------------------
        trijet = w + b1

        good_tops = (
                (trijet.mass > top_low_mass)
                & (trijet.mass < top_up_mass)
        )      

        filtered_tops = trijet.mask[good_tops]
        filtered_ws = w.mask[good_tops]
        
        # ---------------------------------------
        # chi2 criteria
        # ---------------------------------------
        chi2_cal = chi2_test(filtered_tops, filtered_ws, top_sigma, w_sigma, self.top_mass_pdg, self.w_mass_pdg)
        
        good_chi2 = (chi2_cal < chi2)
        tops = filtered_tops.mask[good_chi2]
        
        mask_final = ak.fill_none(good_chi2, False)
        top = ak.fill_none(tops.mass, 0)
        
        
        return top, mask_final
        


    ######################################
    #########  4 Jet #####################
    ######################################
        
    # ----------------------------------------
    #  Scenario IX (b=2  + light_jets = 2)
    # ----------------------------------------      
    # Case 6
    def Scenario_4jets_resolve(self):
        
        top_sigma, top_low_mass, top_up_mass, w_sigma, w_low_mass, w_up_mass, chi2 = tagger_constants("hadronic")
        
        # We select scenarios with 2b and 2jets
        initial_mask = ( 
            (ak.num(self.bjets) == 2)
            & (ak.num(self.jets) == 2)
            & (ak.num(self.fatjets) == 0)
            & (ak.num(self.wjets) == 0)
        )     
        
        mask_objects = ak.fill_none(initial_mask, False)

        
        
        jet_bs = self.bjets.mask[mask_objects]
        jet_jets = self.jets.mask[mask_objects]
        

        jet_b1 = ak.firsts(jet_bs)
        jet_b2 = ak.pad_none(jet_bs, 2)[:, 1]


        jet_l1 = ak.firsts(jet_jets)
        jet_l2 = ak.pad_none(jet_jets, 2)[:, 1]


        # -------------------------------------------
        # Cross cleaning
        cross_cleaning = (
            (jet_b1.delta_r(jet_b2) > self.cross_cleaning)
            & (jet_b1.delta_r(jet_l1) > self.cross_cleaning)
            & (jet_b1.delta_r(jet_l2) > self.cross_cleaning)
            
            & (jet_b2.delta_r(jet_l1) > self.cross_cleaning)
            & (jet_b2.delta_r(jet_l2) > self.cross_cleaning)
            
            & (jet_l1.delta_r(jet_l2) > self.cross_cleaning)            
        )

        b1_cc = jet_b1.mask[cross_cleaning]            
        b2_cc = jet_b2.mask[cross_cleaning]      

        l1_cc = jet_l1.mask[cross_cleaning]            
        l2_cc = jet_l2.mask[cross_cleaning]   


        # ---------------------------------------
        # We will define the W using 2j
        # ---------------------------------------

        dijet_cc = l1_cc + l2_cc

        good_w_mass = (
            (dijet_cc.mass > w_low_mass) 
            & (dijet_cc.mass < w_up_mass) 
        )       

        w = dijet_cc.mask[good_w_mass] 
        b1 = b1_cc.mask[good_w_mass]
        b2 = b2_cc.mask[good_w_mass]


        # ---------------------------------------
        # We will define the top using the b + 2j
        # ---------------------------------------
        condition_top = (
            ((w + b2).mass > top_low_mass) 
            & ((w + b2).mass < top_up_mass)
        )

        trijet = ak.where(
            condition_top,
            w + b2,
            w + b1
        )


        good_tops = (
                (trijet.mass > top_low_mass)
                & (trijet.mass < top_up_mass)
        )      

        
        filtered_tops = trijet.mask[good_tops]
        filtered_ws = w.mask[good_tops]
        
        # ---------------------------------------
        # chi2 criteria
        # ---------------------------------------
        chi2_cal = chi2_test(filtered_tops, filtered_ws, top_sigma, w_sigma, self.top_mass_pdg, self.w_mass_pdg)
        
        good_chi2 = (chi2_cal < chi2)
        tops = filtered_tops.mask[good_chi2]
        
        mask_final = ak.fill_none(good_chi2, False)
        top = ak.fill_none(tops.mass, 0)
        
        
        return top, mask_final



    ######################################
    #########  N Jets ####################
    ######################################
    # fatjets and wjets limitacions have been removed.
    # https://awkward-array.org/doc/main/reference/generated/ak.combinations.html
    # ----------------------------------------
    #  Scenario N (b=2  + light_jets > 2)
    # ----------------------------------------   
    # Case 7
    def Scenario_Njets_resolve(self):
        
        top_sigma, top_low_mass, top_up_mass, w_sigma, w_low_mass, w_up_mass, chi2 = tagger_constants("hadronic")
        
        # We select scenarios with 2b and 2jets
        initial_mask = ( 
            (ak.num(self.bjets) == 2)
            & (ak.num(self.jets) >  2)
        )     
        
        mask_objects = ak.fill_none(initial_mask, False)

        
        
        jet_bs = self.bjets.mask[mask_objects]
        jet_jets = self.jets.mask[mask_objects]
        
     
        jet_b1 = ak.firsts(jet_bs)
        jet_b2 = ak.pad_none(jet_bs, 2)[:, 1]


        # -------------------------------------------
        # Cross cleaning
        cross_cleaning = (
            (jet_b1.delta_r(jet_b2) > self.cross_cleaning)
        )
        
        b1_cc = jet_b1.mask[cross_cleaning]            
        b2_cc = jet_b2.mask[cross_cleaning]      

        jets_cc = jet_jets.mask[cross_cleaning]
                
        
        # W (jj) candidates
        dijet_cc = ak.combinations(jets_cc, 2, fields=["j1", "j2"])
        dijet_cc["p4"] = dijet_cc.j1 + dijet_cc.j2
        
            
        good_w_mass = (
            (dijet_cc["p4"].mass > w_low_mass) 
            & (dijet_cc["p4"].mass < w_up_mass) 
        )       
        
        dijet = dijet_cc["p4"].mask[good_w_mass]
        
        
        # Top (b W) candidates
        dijet_b1 = ak.cartesian({"dijet": dijet, "b1": b1_cc})
        dijet_b2 = ak.cartesian({"dijet": dijet, "b2": b2_cc})
        
        dijet_b1["p4"] = dijet_b1.dijet + dijet_b1.b1
        dijet_b2["p4"] = dijet_b2.dijet + dijet_b2.b2
        
        
        good_b1_top_mass = (
            (dijet_b1["p4"].mass > top_low_mass)
            & (dijet_b1["p4"].mass < top_up_mass)
        )
        
        filtered_dijet_b1 = dijet_b1["p4"].mask[good_b1_top_mass]
        filtered_dijet_1 = dijet.mask[good_b1_top_mass]
        
        
        good_b2_top_mass = (
            (dijet_b2["p4"].mass > top_low_mass)
            & (dijet_b2["p4"].mass < top_up_mass)
        )
        filtered_dijet_b2 = dijet_b2["p4"].mask[good_b2_top_mass]
        filtered_dijet_2 = dijet.mask[good_b2_top_mass]
        
        
        # ---------------------------------------
        # chi2 criteria
        # ---------------------------------------
        chi2_cal_b1 = chi2_test(filtered_dijet_b1, filtered_dijet_1, top_sigma, w_sigma, self.top_mass_pdg, self.w_mass_pdg)
        chi2_1 = ak.fill_none(chi2_cal_b1, 1000)
        
        
        chi2_cal_b2 = chi2_test(filtered_dijet_b2, filtered_dijet_2, top_sigma, w_sigma, self.top_mass_pdg, self.w_mass_pdg)
        chi2_2 = ak.fill_none(chi2_cal_b2, 1000)
        

        # We select tops reconstructed with the lowest chi2.
        chi2_comparison = (chi2_2 < chi2_1)

        chi2_combination = ak.where(
            chi2_comparison,
            chi2_2,
            chi2_1
        )

        filtered_top_combination =  ak.where(
            chi2_comparison,
            filtered_dijet_b2,
            filtered_dijet_b1
        )

        # Min chi2 is selected and arrays are flatten
        min_chi2 = ak.firsts(ak.sort(chi2_combination))
        
        chi2_cal_comb = chi2_combination.mask[chi2_combination == min_chi2]        
        filtered_trijet = filtered_top_combination.mask[chi2_combination == min_chi2]
        

        filtered_chi2_cal = ak.fill_none(ak.sum(chi2_cal_comb, axis = 1), 1000)
        filtered_tops = ak.firsts(ak.sort(filtered_trijet.mass))
        
        
        # Final steps
        good_chi2 = (filtered_chi2_cal < chi2)
        tops = filtered_tops.mask[good_chi2]
        
                
        mask_final = ak.fill_none(good_chi2, False)
        top = ak.fill_none(tops, 0)
        
        
        return top, mask_final
        


    # fatjets and wjets limitacions have been removed.
    # https://awkward-array.org/doc/main/reference/generated/ak.combinations.html
    # ----------------------------------------
    #  Scenario N (b > 2  + light_jets = 2)
    # ----------------------------------------   
    # Case 8
    def Scenario_Nbjets_resolve(self):
        
        top_sigma, top_low_mass, top_up_mass, w_sigma, w_low_mass, w_up_mass, chi2 = tagger_constants("hadronic")
        
        # We select scenarios with 2b and 2jets
        initial_mask = ( 
            (ak.num(self.bjets) > 2)
            & (ak.num(self.jets) ==  2)
        )     
        
        mask_objects = ak.fill_none(initial_mask, False)

        
        
        jet_bs = self.bjets.mask[mask_objects]
        jet_jets = self.jets.mask[mask_objects]
        
        
        jet_l1 = ak.firsts(jet_jets)
        jet_l2 = ak.pad_none(jet_jets, 2)[:, 1]       
        
 
        # -------------------------------------------
        # Cross cleaning
        cross_cleaning = (
            (jet_l1.delta_r(jet_l2) > self.cross_cleaning)
        )
        
        
        l1_cc = jet_l1.mask[cross_cleaning]            
        l2_cc = jet_l2.mask[cross_cleaning]  
        
        bjets_cc = jet_bs.mask[cross_cleaning]  
        
        
        # ---------------------------------------
        # It is define the W using 2j
        # ---------------------------------------

        dijet_cc = l1_cc + l2_cc

        good_w_mass = (
            (dijet_cc.mass > w_low_mass) 
            & (dijet_cc.mass < w_up_mass) 
        )       

        dijets = dijet_cc.mask[good_w_mass] 
        bjets = bjets_cc.mask[good_w_mass]
        
        
        # Combinations between w and 1b are done      
        dijets_bs = ak.cartesian({"w": dijets, "bjet": bjets})
        
        dijets_bs["p4"] = dijets_bs.w + dijets_bs.bjet


        good_b_top_mass = (
            (dijets_bs["p4"].mass > top_low_mass)
            & (dijets_bs["p4"].mass < top_up_mass)
        )
        
        
        filtered_dijets = dijets.mask[good_b_top_mass]
        filtered_trijets_comb = dijets_bs["p4"].mask[good_b_top_mass]
     
    
        chi2_cal_comb = chi2_test(filtered_trijets_comb, filtered_dijets, top_sigma, w_sigma, self.top_mass_pdg, self.w_mass_pdg)
        
        # Min chi2 is selected, and arrays are flatten
        min_chi2 = ak.firsts(ak.sort(chi2_cal_comb))
        
        
        
        chi2_cal = chi2_cal_comb.mask[chi2_cal_comb == min_chi2]
        filtered_trijets = filtered_trijets_comb.mask[chi2_cal_comb == min_chi2]
        
        
        filtered_chi2_cal = ak.fill_none(ak.sum(chi2_cal, axis = 1), 1000)
        filtered_tops = ak.firsts(ak.sort(filtered_trijets.mass))
        
        
        # Final steps
        good_chi2 = (filtered_chi2_cal < chi2)
        tops = filtered_tops.mask[good_chi2]
        
        
        mask_final = ak.fill_none(good_chi2, False)
        top = ak.fill_none(tops, 0)
        
    
        return top, mask_final


    # ---------------------------
    # -- General cases
    # ---------------------------
    # Case 9
    def Scenario_1jet_unresolve_general(self):

        top_sigma, top_low_mass, top_up_mass, w_sigma, w_low_mass, w_up_mass, chi2 = tagger_constants("boosted")


        # We select scenarios with 1 fatjet
        initial_mask = (
            (ak.num(self.bjets) == 0)
            & (ak.num(self.jets) > 0)
            & (ak.num(self.fatjets) == 1)
            & (ak.num(self.wjets) == 0)
        )

        mask_objects = ak.fill_none(initial_mask, False)

        fatjets_masked = self.fatjets.mask[mask_objects]

        # ------
        # Topjet identification
        # ------
        good_top_mass = (
            (fatjets_masked.mass > top_low_mass) 
            & (fatjets_masked.mass < top_up_mass) 
        )

        tops = fatjets_masked.mask[good_top_mass]

        mask_final = ak.fill_none(ak.firsts(good_top_mass), False)
        top = ak.fill_none(ak.firsts(tops.mass), 0)


        return top, mask_final
    
    ######################################
    #########  2 Jet #####################
    ######################################
    # Case 10
    def Scenario_2jets_unresolve_general(self):  
        top_sigma, top_low_mass, top_up_mass, w_sigma, w_low_mass, w_up_mass, chi2 = tagger_constants("boosted")

        # We select scenarios with 1b and 1 fatjet
        initial_mask = (
            (ak.num(self.bjets) == 1)
            & (ak.num(self.jets) > 0)
            & (ak.num(self.fatjets) == 1)
            & (ak.num(self.wjets) == 0)
        )
        
        mask_objects = ak.fill_none(initial_mask, False)
        
        jet_bs = self.bjets.mask[mask_objects]
        jet_fatjets = self.fatjets.mask[mask_objects]
        
        
        # -------------------------------------------
        # Cross cleaning
        cross_cleaning = (
            (jet_bs.delta_r(jet_fatjets) > 2*self.cross_cleaning)
        )

        b1_cc = jet_bs.mask[cross_cleaning]            
        fatjet_cc = jet_fatjets.mask[cross_cleaning]            

        
        # ------
        # Topjet identification
        # ------
        good_top_mass = (
            (fatjet_cc.mass > top_low_mass) 
            & (fatjet_cc.mass < top_up_mass) 
        )
        
        
        tops = fatjet_cc.mask[good_top_mass]
        mask_final = ak.fill_none(ak.firsts(good_top_mass), False)
        top = ak.fill_none(ak.firsts(tops.mass), 0)
        
 
        return top, mask_final
    
    
    # ----------------------------------------
    #  Scenario V (W = 1  +  b = 1)
    # ----------------------------------------
    # Case 11
    def Scenario_2jets_partiallyresolve_general(self):
        
        top_sigma, top_low_mass, top_up_mass, w_sigma, w_low_mass, w_up_mass, chi2 = tagger_constants("Partially_boosted")
        
        # We select scenarios with 1b and 1 wjet
        initial_mask = (
            (ak.num(self.bjets) == 1)
            & (ak.num(self.jets) > 0)
            & (ak.num(self.fatjets) == 0)
            & (ak.num(self.wjets) == 1)
        )
        
        mask_objects = ak.fill_none(initial_mask, False)
        
        jet_bs = self.bjets.mask[mask_objects]
        jet_wjets = self.wjets.mask[mask_objects]
        
        jet_b1 = ak.firsts(jet_bs)
        jet_w1 = ak.firsts(jet_wjets)
        
        # -------------------------------------------
        # Cross cleaning
        cross_cleaning = (
            (jet_b1.delta_r(jet_w1) > 2*self.cross_cleaning)           
        )

        b1_cc = jet_b1.mask[cross_cleaning]            
        w1_cc = jet_w1.mask[cross_cleaning]            

        
        # ------
        # Wjet identification
        # ------
        good_w_mass = (
            (w1_cc.mass > w_low_mass) 
            & (w1_cc.mass < w_up_mass) 
        )
        
        w = w1_cc.mask[good_w_mass]
        b1 = b1_cc.mask[good_w_mass]

        
        # ---------------------------------------
        # We will define the top using the b + w
        # ---------------------------------------
        dijet= w + b1
        
        good_tops = (
                (dijet.mass > top_low_mass)
                & (dijet.mass <  top_up_mass)
        )   
        
        filtered_tops = dijet.mask[good_tops]
        filtered_ws = w.mask[good_tops]
        
        # ---------------------------------------
        # chi2 criteria
        # ---------------------------------------
        chi2_cal = chi2_test(filtered_tops, filtered_ws, top_sigma, w_sigma, self.top_mass_pdg, self.w_mass_pdg)
    
        
        
        good_chi2 = (chi2_cal < chi2)
        tops = filtered_tops.mask[good_chi2]
        
        mask_final = ak.fill_none(good_chi2, False)
        top = ak.fill_none(tops.mass, 0)
        
        
        return top, mask_final
    
    
    
    ######################################
    #########  3 Jet #####################
    ######################################
        
    # ----------------------------------------
    #  Scenario V (W = 1  +  b = 2)
    # ----------------------------------------
    # Case 12
    def Scenario_3jets_partiallyresolve_general(self):
        
        top_sigma, top_low_mass, top_up_mass, w_sigma, w_low_mass, w_up_mass, chi2 = tagger_constants("Partially_boosted")
        
        # We select scenarios with 2b and 1 wjet
        initial_mask = (
            (ak.num(self.bjets) == 2)
            & (ak.num(self.jets) == 0)
            & (ak.num(self.fatjets) > 0)
            & (ak.num(self.wjets) == 1)
        )
        
        mask_objects = ak.fill_none(initial_mask, False)
        
        jet_bs = self.bjets.mask[mask_objects]
        jet_wjets = self.wjets.mask[mask_objects]
        
        jet_b1 = ak.firsts(jet_bs)
        jet_b2 = ak.pad_none(jet_bs, 2)[:, 1]


        jet_w1 = ak.firsts(jet_wjets)
        
        # -------------------------------------------
        # Cross cleaning
        cross_cleaning = (
            (jet_b1.delta_r(jet_b2) > self.cross_cleaning)
            & (jet_b1.delta_r(jet_w1) > 2*self.cross_cleaning)
            & (jet_b2.delta_r(jet_w1) > 2*self.cross_cleaning)
        )

        b1_cc = jet_b1.mask[cross_cleaning]            
        b2_cc = jet_b2.mask[cross_cleaning]      

        w1_cc = jet_w1.mask[cross_cleaning]            

        
        # ------
        # Wjet identification
        # ------
        good_w_mass = (
            (w1_cc.mass > w_low_mass) 
            & (w1_cc.mass < w_up_mass) 
        )
        
        w = w1_cc.mask[good_w_mass]
        b1 = b1_cc.mask[good_w_mass]
        b2 = b2_cc.mask[good_w_mass]
        
        # ---------------------------------------
        # We will define the top using the b + w
        # ---------------------------------------
        condition = (
            ((w + b1).mass > top_low_mass) 
            & ((w + b1).mass < top_up_mass)
        )
        
        dijet = ak.where(condition,
                       w + b1,
                       w + b2
        )

        good_tops = (
                (dijet.mass > top_low_mass)
                & (dijet.mass <  top_up_mass)
        )   
        
        filtered_tops = dijet.mask[good_tops]
        filtered_ws = w.mask[good_tops]
        
        
        # ---------------------------------------
        # chi2 criteria
        # ---------------------------------------
        chi2_cal = chi2_test(filtered_tops, filtered_ws,  top_sigma, w_sigma, self.top_mass_pdg, self.w_mass_pdg)

        
        good_chi2 = (chi2_cal < chi2)
        tops = filtered_tops.mask[good_chi2]
        
        mask_final = ak.fill_none(good_chi2, False)
        top = ak.fill_none(tops.mass, 0)
        
        return top, mask_final
    
    # Case 13
    def Scenario_3jets_resolve_general(self):
        
        top_sigma, top_low_mass, top_up_mass, w_sigma, w_low_mass, w_up_mass, chi2 = tagger_constants("hadronic")
        
        # We select scenarios with 1b and 2jets
        initial_mask = ( 
            (ak.num(self.bjets) == 1)
            & (ak.num(self.jets) > 2)
            & (ak.num(self.fatjets) == 0)
            & (ak.num(self.wjets) == 0)
        )     
        
        mask_objects = ak.fill_none(initial_mask, False)

        
        
        jet_b = ak.firsts(self.bjets.mask[mask_objects])
        jet_jets = self.jets.mask[mask_objects]
        

  
        # W (jj) candidates
        dijets = ak.combinations(jet_jets, 2, fields=["j1", "j2"])
        dijets["p4"] = dijets.j1 + dijets.j2

        
        good_w_mass = (
            (dijets["p4"].mass > top_tagger_mW_mTop_Njets_selection[self.lepton_flavor]["m_W_min"]) 
            & (dijets["p4"].mass < top_tagger_mW_mTop_Njets_selection[self.lepton_flavor]["m_W_max"]) 
        )       
        
        ws = dijets["p4"].mask[good_w_mass]  
        
        bs = jet_b.mask[good_w_mass]
        
        
        # Top (b W) candidates
        #dijet_b = ak.cartesian({"dijet": ws, "b": bs})
        #dijet_b["p4"] = dijet_b.dijet + dijet_b.b

        trijet = ws + bs
        
        good_top_mass = (
            (trijet.mass > top_tagger_mW_mTop_Njets_selection[self.lepton_flavor]["m_Top_min"])
            & (trijet.mass < top_tagger_mW_mTop_Njets_selection[self.lepton_flavor]["m_Top_max"])
        )


        filtered_trijet = trijet.mask[good_top_mass]
        filtered_dijet = ws.mask[good_top_mass]
        
        
        # ---------------------------------------
        # chi2 criteria
        # ---------------------------------------
        chi2_cal_b = chi2_test(filtered_trijet, filtered_dijet, top_sigma, w_sigma, self.top_mass_pdg, self.w_mass_pdg)
        chi2_combinations = ak.fill_none(chi2_cal_b, 1000)
        
        
        # Min chi2 is selected and arrays are flatten
        min_chi2 = ak.firsts(ak.sort(chi2_combinations))

        chi2_cal = chi2_combinations.mask[chi2_combinations == min_chi2]     
        filtered_trijet_comb = filtered_trijet[chi2_combinations == min_chi2]
        
        
        filtered_chi2_cal = ak.fill_none(ak.sum(chi2_cal, axis = 1), 1000)
        filtered_tops = ak.firsts(ak.sort(filtered_trijet_comb.mass))
        
        # Final steps
        good_chi2 = (filtered_chi2_cal < chi2)
        tops = filtered_tops.mask[good_chi2]
        
                
        mask_final = ak.fill_none(good_chi2, False)
        top = ak.fill_none(tops, 0)
        
        
        return top, mask_final  

 