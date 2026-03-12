import re
import json
import hist
import numpy as np
import correctionlib
import awkward as ak
from typing import Type
from coffea import util
import importlib.resources
from coffea.analysis_tools import Weights
from coffea.lookup_tools.dense_lookup import dense_lookup

from wprime_plus_b.corrections.utils import get_pog_json
    
    
class BTagCorrector:
    def __init__(
        self, 
        jets: ak.Array,
        bjet_mask: ak.Array,
        mask_eff_btag: ak.Array,    
        weights: Type[Weights],
        dataset: str = "",
        year: str = "2017",
        tagger: str = "deepJet",
        pass_working_point: str = "Tight",
        fail_working_point: str = None
    ):
        """
        Ref: https://twiki.cern.ch/twiki/bin/view/CMS/BTagSFMethods
        
        """
        # =========================================================
        #     Load efficiency lookup table (only for deepJet)
        #             efflookup(pt, |eta|, flavor)
        # =========================================================        
        self.year = year
        self.tagger = tagger
        self.dataset_name = re.sub(r'_\d+$', '', dataset)

        self.mask_eff_btag = mask_eff_btag
        
        self.bjet_mask = bjet_mask
        self.weights = weights

        # ==========================================================
        #  Load btag working point
        # ==========================================================
        with open("wprime_plus_b/json_files/bjet.json", "r") as f:
            btag_wps = json.load(f)[tagger][year]
        
        self.pass_btag_wp = btag_wps[pass_working_point]
        self.pass_wp = pass_working_point
            
        if fail_working_point is not None:
            self.fail_btag_wp = btag_wps[fail_working_point]
            self.fail_wp = fail_working_point
            self.multiple_wps = True            
        else:
            self.multiple_wps = False


        # ===========================================================
        #  Correction 
        # ============================================================
        # Correction name
        with open("wprime_plus_b/corrections/correction_names/BTV.json", "r") as f:
            case = json.load(f)

        self.correction_name_map = {
            "bc": case["CMS_btag_fixedWP_bc_simple"][year],
            "light": case["CMS_btag_fixedWP_light_simple"][year]
        }
        # define correction set
        self.cset = correctionlib.CorrectionSet.from_file(get_pog_json(json_name="btag", year=year))

        # ===========================================================
        #  Flavor: 5=b, 4=c, 0=udsg
        # ============================================================
        self.jet_map = {
            "bc": jets[jets.hadronFlavour > 0],
            "light": jets[jets.hadronFlavour == 0]
        }      

    def get_eff(self, jets_pt, abs_jets_eta, jets_btag, jets_flavor, btag_wp_value, flavor):
        """
        https://btv-wiki.docs.cern.ch/ScaleFactors/
        https://btv-wiki.docs.cern.ch/PerformanceCalibration/fixedWPSFRecommendations/#b-tagging-efficiencies-in-simulation

        Tagging efficiencies depend on event kinematics.  Therefore, it is mandatory to first compute from simulation the tagging efficiencies for each flavour as a function of the jet 
        pT and jet |eta|  for your specific analysis. Afterwards, you can weigh them by the provided SF values using one of the recommendations linked in the table below.

        -  pT: A recommended binning is along the same lines as is used in the scale factor derivation, e.g. [20, 30, 50, 70, 100, 140, 200, 300, 600, 1000].
        -  η: A recommended binning depends on the available simulation statistics, one could e.g. only use one single bin, or separate into barrel and endcap regions, or use a few homogeneous bins.

        Please note that the derivation of these efficiency maps only makes sense if no b-tagging related event selections have been applied yet. Best practice is to derive these efficiencies after the analysis 
        event selection, but omitting any b-tagging selection, such as the number of b-tagged jets, etc.

        """
        # Jets passing the btag wp
        is_btagged = jets_btag > btag_wp_value

        # Flavor mask: 5: b, 4: c, 0: light (udsg)
        flavor_masks = {
            "b": (jets_flavor == 5),
            "c": (jets_flavor == 4),
            "bc": (jets_flavor == 4) | (jets_flavor == 5),
            "light": (jets_flavor == 0),
        }

        pt_bins = [20, 30, 50, 70, 100, 150, 200, 300, 500, 1000]
        eta_bins = [0.0, 1.3, 2.5, 3.0, 5.2]

        def make_h2():
            return hist.Hist(
                hist.axis.Variable(pt_bins, name="pt"),
                hist.axis.Variable(eta_bins, name="eta"),
            )

        flavor_mask = flavor_masks[flavor]
        tagged_mask = flavor_mask & is_btagged


        # Denominator: Only flavor mask
        h_denom = make_h2()
        h_denom.fill(
            pt=jets_pt[flavor_mask],
            eta=abs_jets_eta[flavor_mask]
        )

        # Numerator: Flavor mask & btag mask
        h_num = make_h2()
        h_num.fill(
            pt=jets_pt[tagged_mask],
            eta=abs_jets_eta[tagged_mask]
        )

        num = h_num.values()
        den = h_denom.values()

        eff = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

        eff_lookup = dense_lookup(eff, [ax.edges for ax in h_denom.axes])

        return eff_lookup

        

    def add_btag_weights(self, flavor: str, correlated: bool):

        # --------------------------------------------------------
        #  1. Jet candidates
        # --------------------------------------------------------
        jets = self.jet_map[flavor]
        j, nj = ak.flatten(jets), ak.num(jets)

        
        jet_eta_mask = (np.abs(j.eta) < 2.5) 
        jet_mask = jet_eta_mask
        in_jets = j.mask[jet_mask]
        
        # get jet pt and eta (replace None values with some 'in-limit' value)
        jet_pt = ak.fill_none(in_jets.pt, 0.0)
        jet_eta = ak.fill_none(np.abs(in_jets.eta), 0.0)
        
        
        
        # ---------------------------------------------------------
        # 2. Efficiencies (MC)
        # ---------------------------------------------------------
        # Important: If two wps are considered, e.g Loose no Medium, the pass variable will be looser than the fail variable.
        # Use jets passing general cuts without criteria associated with the number of jets.
        #jets_eff = j[self.mask_eff_btag]
        jets_eff = ak.flatten(jets[self.mask_eff_btag])

        # Bjets passing wp
        eff_lookup_p = self.get_eff(
            jets_pt=jets_eff.pt, 
            abs_jets_eta=np.abs(jets_eff.eta), 
            jets_btag=jets_eff.btagDeepFlavB,
            jets_flavor=jets_eff.hadronFlavour, 
            btag_wp_value=self.pass_btag_wp,
            flavor=flavor
        )
        # Evaluate
        eff_p = eff_lookup_p(jets.pt, np.abs(jets.eta))
        
        # Bjets failing wp
        eff_lookup_f = self.get_eff(
            jets_pt=jets_eff.pt, 
            abs_jets_eta=np.abs(jets_eff.eta), 
            jets_btag=jets_eff.btagDeepFlavB,
            jets_flavor=jets_eff.hadronFlavour, 
            btag_wp_value=self.fail_btag_wp,
            flavor=flavor
        ) if self.multiple_wps else None
        # Evaluate
        eff_f = eff_lookup_f(jets.pt, np.abs(jets.eta)) if self.multiple_wps else None
               
        
        # Define whether it is a correlated case or not
        suffix = "_correlated" if correlated else ""
        variations = {"nominal": "central", "up": f"up{suffix}", "down": f"down{suffix}"}


        # -----------------------------------------------------------
        # 3. Get the correction name
        # -----------------------------------------------------------
        correction_name = self.correction_name_map[flavor]
        
        weights = {}
        for label, var in variations.items():
            # var: "nominal", "up", "down
            # -----------------------------------------------------------
            # 4. Obtain the scale factor
            # -----------------------------------------------------------
            # Read the value of the scale factor
            sf_evaluated = self.cset[correction_name].evaluate(
                var, self.pass_wp[0], j.hadronFlavour, np.abs(jet_eta), jet_pt
            )
            
            # Discarding artificial values
            sf_masked = ak.where(jet_mask, sf_evaluated, 1.0)

            # Restore the original structure per event.
            sf_p = ak.unflatten(sf_masked, nj)
            

            if not self.multiple_wps:
                # -----------------------------------------------------------
                # 5. Calculate the weight, case of 1 wp.
                # -----------------------------------------------------------
                passbtag = (jets.btagDeepFlavB > self.pass_btag_wp)
                weight = ak.where(passbtag, sf_p, (1 - sf_p * eff_p) / (1 - eff_p))


                den = 1 - eff_p
                mask_bad = den == 0

                if ak.any(mask_bad):
                    print("⚠️ eff_p = 1 encontrado en", ak.sum(mask_bad), "jets de un número total de ", len(mask_bad))

            else:
                # -----------------------------------------------------------
                # 5. Calculate the SF of the second working point
                # -----------------------------------------------------------
                sf_evaluated = self.cset[correction_name].evaluate(
                    var, self.fail_wp[0], j.hadronFlavour, np.abs(jet_eta), jet_pt
                )
                
                # Discarding artificial values
                sf_masked = ak.where(jet_mask, sf_evaluated, 1.0)
    
                # Restore the original structure per event.
                sf_f = ak.unflatten(sf_masked, nj)

                # -----------------------------------------------------------
                # 6. Masks of the three terms
                # -----------------------------------------------------------
                # Jets passing the tightest WP
                mask_pass_tightest = (jets.btagDeepFlavB > self.fail_btag_wp)

                # Jets within the band defined by the two wps
                mask_in_interval = (
                    (jets.btagDeepFlavB > self.pass_btag_wp) 
                    & (jets.btagDeepFlavB <= self.fail_btag_wp)
                )

                # Jet fails the softest wp
                mask_fail_all = (jets.btagDeepFlavB <= self.pass_btag_wp)

                # -----------------------------------------------------------
                # 7. Calculation of the three terms of the product separately
                # -----------------------------------------------------------
                # Category: i Tagged T 
                first_term = sf_f 

                # Category: Tagged L not T
                delta_eff = eff_p - eff_f
                second_term = (eff_p * sf_p - eff_f * sf_f) / (eff_p - eff_f)
                

                # Category: No tagged.
                third_term = (1 - eff_p * sf_p) / (1 - eff_p)

                # -----------------------------------------------------------
                # 6. Calculate the weights for the case of 2 wps.
                # -----------------------------------------------------------

                weight = ak.ones_like(jets.pt)

                # Assign the corresponding term to each jet according to its mask
                weight = ak.where(mask_pass_tightest, first_term, weight)
                weight = ak.where(mask_in_interval, second_term, weight)
                weight = ak.where(mask_fail_all, third_term, weight)
                

            # The weight of the event is the product of the weights of all its jets.    
            weights[label] = ak.fill_none(ak.prod(weight, axis=-1), 1.0)


        
        """
        nominal_sf, up_sf, down_sf = [
            ak.where(self.bjet_mask, sf, 1.0)
            for sf in (weights["nominal"], weights["up"], weights["down"])
        ]  
        """
        nominal_sf, up_sf, down_sf = [
            ak.where(self.bjet_mask, ak.nan_to_num(sf, nan=1.0, posinf=1.0, neginf=1.0), 1.0)
            for sf in (weights["nominal"], weights["up"], weights["down"])
        ]

            
        # add scale factors to weights container
        self.weights.add(
            name=f"CMS_btag_{'heavy' if flavor == 'bc' else 'light'}_{self.year}",
            weight=nominal_sf,
            weightUp=up_sf,
            weightDown=down_sf
        )