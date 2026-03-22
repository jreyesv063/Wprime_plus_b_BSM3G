import json
import copy
import correctionlib
import numpy as np
import awkward as ak
import importlib.resources
from typing import Type
from pathlib import Path
from typing import Optional, Union, List
from coffea.analysis_tools import Weights
from wprime_plus_b.corrections.utils import pog_years, get_pog_json, unflat_sf


"""
TauID corrections

https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun2
https://github.com/cms-tau-pog/TauIDSFs
https://github.com/uhh-cms/hh2bbtautau/blob/7666ed0426c87baa8d143ec26a216c3cadde513b/hbt/calibration/tau.py#L60

Good example:
https://github.com/schaefes/hh2bbtautau/blob/da6d47a7ddb2b1e7ffda06b8a96c6ddead2824b8/hbt/production/tau.py#L108

# genuine taus
* DeepTau2017v2p1VSjet (pt [-inf, inf); dm (0, 1, 2, 10, 11) ; genmatch (0, 1, 2, 3, 4, 5, 6 ); wp (Loose, Medium, Tight, VTight); wp_VSe (Tight, VVLoose );  syst ; flag (dm, pt))


# electrons faking taus
* DeepTau2017v2p1VSe = eta [0.0, 2.3); genmatch (0, 1); wp (Loose, Medium, Tight, VLoose, VTight, VVLoose, VVTight; syst (down, nom, up)

# muons faking taus
* DeepTau2017v2p1VSmu (eta [0.0, 2.3); genmatch (0, 2); wp (Loose, Medium, Tight, VLoose); syst (down, nom, up))

        
"""


class TauCorrector:
    def __init__(
        self,
        taus: ak.Array,
        weights: Type[Weights],
        year: str = "2017",
        tau_vs_jet: str = "Tight",
        tau_vs_ele: str = "Tight",
        tau_vs_mu: str = "Tight",
        tau_mask = None
    ) -> None:


        self.tau_mask = tau_mask
        self.weights = weights
        self.year = year
        
        # Working points
        tau_id_version = "DeepTau2017" if self.year in ["2016APV", "2016", "2017", "2018"] else "DeepTau2018"        
        self.suffix = "2017v2p1" if year in ["2016APV", "2016", "2017", "2018"] else "2018v2p5"
        
        # ============================================================
        # Load tau information
        # ============================================================
        with open("wprime_plus_b/json_files/tau.json", "r") as f:
            taus_info = json.load(f)

       
        
        self.tau_vs_jet_wp = taus_info[tau_id_version]["tau_vs_jet"][tau_vs_jet]
        self.tau_vs_ele_wp = taus_info[tau_id_version]["tau_vs_e"][tau_vs_ele]
        self.tau_vs_mu_wp = taus_info[tau_id_version]["tau_vs_mu"][tau_vs_mu]

        self.taus_genMatch = taus_info["genPartFlav"] 
        self.taus_prongs = taus_info["prongs"] 
        

        # flat taus array
        self.taus, self.n = ak.flatten(taus), ak.num(taus)

        # tau transverse momentum, pseudorapidity, genPartFlav and decayMode
        self.taus_pt = self.taus.pt
        self.taus_eta = self.taus.eta
        self.taus_genPartFlav = self.taus.genPartFlav
        self.taus_decayMode = self.taus.decayMode
       
        
        self.taus_wp_jet = getattr(self.taus, f"idDeepTau{self.suffix}VSjet")
        self.taus_wp_e = getattr(self.taus, f"idDeepTau{self.suffix}VSe")  
        self.taus_wp_mu = getattr(self.taus, f"idDeepTau{self.suffix}VSmu")


        # DeepTau working points
        self.tau_vs_jet = tau_vs_jet
        self.tau_vs_ele = tau_vs_ele
        self.tau_vs_mu = tau_vs_mu



        # define correction set_id
        self.cset = correctionlib.CorrectionSet.from_file(get_pog_json(json_name="tau", year=self.year))
        self.pog_year = pog_years[year]

        # ===========================================================
        #  Read json file: corrections
        # ============================================================
        # Correction name
        with open("wprime_plus_b/corrections/correction_names/TAU.json", "r") as f:
            self.case = json.load(f)
            
        """
        Check: https://github.com/cms-tau-pog/TauFW/blob/43bc39474b689d9712107d53a953b38c3cd9d43e/PicoProducer/python/analysis/ModuleETau.py#L270 
        """

    # e -> tau_h fake rate SFs for DeepTau2017v2p1VSe
    # eta = (0, 2.3]; genMatch = 0,1; wp = Loose, Medium, Tight, VLoose, VTight, VVLoose, VVTight; syst: down, nom, up

    def add_id_weight_DeepTau2017v2p1VSe(self):
        """
        Sf is called with:

        evaluate(eta (real),  genmatch (int) , wp (string), syst (string))

        """

        correction_name = self.case["CMS_fake_t_DeepTau_VSe"][self.year]

        # =============================================================
        #  Tau candidates
        # =============================================================           
        # tau pseudorapidity range: [0, 2.3)
        tau_eta_mask = np.abs(self.taus_eta < 2.3) 
        # GenMatch = 0 "unmatched", 1 "electron";
        tau_genMatch_mask = (self.taus_genPartFlav == self.taus_genMatch["prompt_electron"]) | (self.taus_genPartFlav == self.taus_genMatch["tau_e_decay"])
        # Only taus passing the wp stablished
        tau_wp_mask = (self.taus_wp_e >= self.tau_vs_ele_wp)

        in_tau_mask = tau_genMatch_mask & tau_wp_mask  & tau_eta_mask
        # get 'in-limits' taus
        in_limit_taus = self.taus.mask[in_tau_mask]

        
        # fill Nones with some 'in-limit' value
        tau_eta = ak.fill_none(in_limit_taus.eta, 0)
        tau_genMatch = ak.fill_none(in_limit_taus.genPartFlav, 0.0)


        # =============================================================
        # Correction: event-level weight (nominal/up/down)
        # =============================================================        
        # Get nominal, up, and down scale factors
        nominal_sf, up_sf, down_sf = [
            unflat_sf(self.cset[correction_name].evaluate(tau_eta, tau_genMatch, self.tau_vs_ele, v), in_tau_mask, self.n)
            for v in ("nom", "up", "down")
        ]


        nominal_sf, up_sf, down_sf = [
            ak.where(self.tau_mask, sf, 1.0)
            for sf in (nominal_sf, up_sf, down_sf)
        ]

        # add scale factors to weights container
        self.weights.add(
            name=f"CMS_fake_t_DeepTau{self.suffix}_VSe_{self.year}",
            weight=nominal_sf,
            weightUp=up_sf,
            weightDown=down_sf,
        )

        return nominal_sf

    # mu -> tau_h fake rate SFs for DeepTau2017v2p1VSmu
    # eta = (0, 2.3]; genMatch = 0,2; wp = Loose, Medium, Tight, VLoose ; syst: down, nom, up

    def add_id_weight_DeepTau2017v2p1VSmu(self):
        """
        Sf is called with:

        evaluate(eta (real),  genmatch (int) , wp (string), syst (string))

        """

        correction_name = self.case["CMS_fake_t_DeepTau_VSmu"][self.year]
        
        # =============================================================
        #  Tau candidates
        # =============================================================     
        # tau pseudorapidity range: [0, 2.3)
        tau_eta_mask = np.abs(self.taus_eta < 2.3) 
        # GenMatch = 0 "unmatched", 2 "muon";
        tau_genMatch_mask = (self.taus_genPartFlav == self.taus_genMatch["prompt_muon"]) | (self.taus_genPartFlav == self.taus_genMatch["tau_mu_decay"])
        
        # Only taus passing the wp stablished
        tau_wp_mask = (self.taus_wp_mu >= self.tau_vs_mu_wp)
        in_tau_mask = tau_genMatch_mask & tau_wp_mask  & tau_eta_mask 
        # get 'in-limits' taus
        in_limit_taus = self.taus.mask[in_tau_mask]

        # fill Nones with some 'in-limit' value
        tau_eta = ak.fill_none(in_limit_taus.eta, 0)
        tau_genMatch = ak.fill_none(in_limit_taus.genPartFlav, 0.0)


        # =============================================================
        # Correction: event-level weight (nominal/up/down)
        # =============================================================        
        # Get nominal, up, and down scale factors
        nominal_sf, up_sf, down_sf = [
            unflat_sf(self.cset[correction_name].evaluate(tau_eta, tau_genMatch, self.tau_vs_mu, v), in_tau_mask, self.n)
            for v in ("nom", "up", "down")
        ]


        nominal_sf, up_sf, down_sf = [
            ak.where(self.tau_mask, sf, 1.0)
            for sf in (nominal_sf, up_sf, down_sf)
        ]        
        
        # add scale factors to weights container
        self.weights.add(
            name=f"CMS_fake_t_DeepTau{self.suffix}_VSmu_{self.year}",
            weight=nominal_sf,
            weightUp=up_sf,
            weightDown=down_sf,
        )



    def add_id_weight_DeepTau2017v2p1VSjet(
        self, 
        flag: str = "pt"
        ):
        """
        https://github.com/LEAF-HQ/LEAF/blob/d22cc55594a4b16d061c25dbf7ecdec04eedbc34/Analyzer/src/TauScaleFactorApplicatorJson.cc#L28

        Sf is called with:

        evaluate(pt (real),  dm (int), genmatch (int), wp (string), wp_VSe (string), syst (string), flag (string))

         - dm (decay mode): 0 (tau->pi); 1 (tau->rho->pi+pi0); 2 (tau->a1->pi+2pi0); 10 (tau->a1->3pi); 11 (tau->3pi+pi0)
         - getmatch: 0 or 6 = unmatched or jet, 1 or 3 = electron, 2 or 4 = muon, 5 = real tau
         - flag: We have worked in 'pt' = pT-dependent

        By default, use the pT-dependent SFs with the 'pt' flag
        pt = (-inf, inf); dm = 0, 1, 2, 10, 11; genmatch = 0, 1, 2, 3, 4, 5, 6; wp = Loose, Medium, Tight, VTight; wp_VSe = Tight, VVLoose; syst = down, nom, up; flag = dm, pt
        
        """

        correction_name = self.case["CMS_fake_t_DeepTau_VSjet"][self.year]
        
        # =============================================================
        #  Tau candidates
        # =============================================================         
        # tau decayMode
        tau_dm_mask = ak.zeros_like(self.taus_decayMode, dtype=bool)
        for decay_mode in self.taus_prongs["1or3prongs"]:
            tau_dm_mask = tau_dm_mask | (self.taus_decayMode == decay_mode)

        # GenMatch = 0 or 6 = unmatched or jet, 1 or 3 = electron, 2 or 4 = muon, 5 = real tau
        tau_genMatch_mask = (self.taus_genPartFlav == self.taus_genMatch["hadronic_tau_decay"])
        
        # Only taus passing the wp stablished
        tau_wp_mask = (
            (self.taus_wp_jet >= self.tau_vs_jet_wp)  # vs Jet mask
            & (self.taus_wp_jet >= self.tau_vs_ele_wp) # vs Ele mask
        )
        in_tau_mask = tau_dm_mask & tau_genMatch_mask & tau_wp_mask
        # get 'in-limits' taus
        in_limit_taus = self.taus.mask[in_tau_mask]
        # get pt and eta
        # fill Nones with some 'in-limit' value
        tau_pt = ak.fill_none(in_limit_taus.pt, 0)
        tau_dm = ak.fill_none(in_limit_taus.decayMode, 0)
        tau_genMatch = ak.fill_none(in_limit_taus.genPartFlav, 0.0)


        # =============================================================
        # Correction: event-level weight (nominal/up/down)
        # =============================================================     
        # Tau ID correction is only available for Loose, Medium, Tight, and VTight
        wp_jet = {
            "VVVLoose": "Loose",
            "VVLoose": "Loose",
            "VLoose": "Loose",
            "Loose": "Loose",
            "Medium": "Medium",
            "Tight": "Tight",
            "VTight": "VTight"
        }   
        # Get nominal, up, and down scale factors
        nominal_sf, up_sf, down_sf = [
            unflat_sf(self.cset[correction_name].evaluate(tau_pt, tau_dm, tau_genMatch, wp_jet[self.tau_vs_jet], self.tau_vs_ele, v, flag), in_tau_mask, self.n)
            for v in ("default", "up", "down")
        ]

        
        nominal_sf, up_sf, down_sf = [
            ak.where(self.tau_mask, sf, 1.0)
            for sf in (nominal_sf, up_sf, down_sf)
        ]
        
        # add scale factors to weights container
        self.weights.add(
            name=f"CMS_fake_t_DeepTau{self.suffix}_VSjet_{self.year}",
            weight=nominal_sf,
            weightUp=up_sf,
            weightDown=down_sf,
        )
        

    #def add_id_weight_diTauTrigger(self, mask_trigger, trigger: str = "ditau", info: str = "sf", dm: int = -1, trigger_name: str | list[str] = "HLT", events: ak.Array | None = None):
    def add_ditau_trigger_weight(
            self, 
            trigger_mask,
            trigger_match_mask,
            trigger: str = "ditau", 
            info: str = "sf", 
            dm: int = -1
        ):
        """
        Computes and stores CMS Tau Trigger scale factors (nominal/up/down) for diTau-based triggers.
        
        Tau Trigger SFs and efficiencies for {0} ditau, etau, mutau or ditauvbf triggers. Ditauvbf
        
        trigger SF is only available for 2017 and 2018. To get the usual DM-specific SF's, specify the
        DM, otherwise set DM to -1 to get the inclusive SFs. Default corrections are set to SF's, if you
        require the input efficiencies, you can specify so in the corrtype input variable

        pt = [24.59953, inf); dm = -1, 0, 1, 10; trigtype = 'ditau', 'etau', 'mutau', 'ditauvbf; wp "DeepTauVSjet"= Loose, Medium, Tight, VLoose, VTight, VVLoose, VVTight, VVVLoose; corrtype =  eff_data, eff_mc, sf;  syst = down, nom, up


        Run 2: https://github.com/cms-tau-pog/TauTriggerSFs/tree/run2_SFs
        

        """

        correction_name = self.case["CMS_trig_t_ditau"][self.year]

        # =============================================================
        # Trigger mask: Only events that pass the selected trigger
        # =============================================================
        masks = [events.HLT[trig] for trig in trigger_name if trig in events.HLT.fields]
        if masks:
            mask_reference_trigger = ak.any(ak.Array(masks), axis=0)
        else:
            # If no matching triggers, return all False
            mask_reference_trigger = ak.zeros(len(events), dtype=bool)
        
        
        # =============================================================
        #  Tau candidates
        # =============================================================            
        # tau pt range: [24.59953, inf]
        tau_pt_mask = (self.taus_pt >= 40)

        # tau decayMode
        for decay_mode in self.taus_prongs["trigger_corr"]:
            tau_dm_mask= tau_dm_mask | (self.taus_decayMode == decay_mode)
            
        # Only taus passing the wp stablished
        tau_wp_mask = self.taus_wp_jet > self.tau_vs_jet_wp
        tau_mask = tau_pt_mask & tau_dm_mask & tau_wp_mask
        # get 'in-limits' taus
        in_limit_taus = self.taus.mask[tau_mask]
        # get pt and dm
        # fill Nones with some 'in-limit' value
        tau_pt = ak.fill_none(in_limit_taus.pt, 40.0)
        tau_dm = ak.fill_none(in_limit_taus.decayMode, -1)
        trigtype = trigger
        corrtype = info


        # =============================================================
        # Correction: event-level weight (nominal/up/down)
        # =============================================================  
        # Get nominal, up, and down scale factors
        nominal_sf, up_sf, down_sf = [
            unflat_sf(self.cset[correction_name].evaluate(tau_pt, tau_dm, trigtype, self.tau_vs_jet, v), tau_mask, self.n)
            for v in ("nom", "up", "down")
        ]

        # add scale factors to weights container
        self.weights.add(
            name=f"CMS_trig_t_ditau_{self.tau_vs_jet}_eff_mc_{self.year}",
            weight=nominal_sf,
            weightUp=up_sf,
            weightDown=down_sf,
        )
