import re
import numpy as np
import awkward as ak
from copy import copy
from coffea.analysis_tools import PackedSelection, Weights


# =======================================
#  Plots
# =======================================
from wprime_plus_b.processors.utils.histograms import Histograms

# Utils
from wprime_plus_b.processors.utils.analysis_utils import check_object_cut_dependency, cross_cleaning, fill_cutflow, map_object_level_var

# Top tagger
from wprime_plus_b.processors.utils.utils_topXfinder import get_topXfinder_masks


class Systematics:
    def __init__(
        self,
        cc,
        cr,
        year,
        weights,
        objects,
        criteria,
        metadata,
        cut_names,
        processor,
        histograms,
        selections,
        table_name,
        variations,
        lepton_flavor    
    ):

        """
        Important: the MET has mapped all systematic variations made.
        """
        self.cr = cr
        self.cc = cc
        self.year = year
        self.objects = objects
        self.weights = weights
        self.criteria = criteria        
        self.metadata = metadata  
        self.cut_names = cut_names
        self.processor = processor
        self.histograms = histograms
        self.selections = selections
        self.table_name = table_name
        self.variations = variations
        self.lepton_flavor = lepton_flavor

        if self.cr is not None:
            self.invert_delta_phi = self.criteria["data_driven_qcd_estimation"][self.cr][self.lepton_flavor]["invert_delta_phi"]
        
        else:
            self.invert_delta_phi = self.criteria["met"][self.lepton_flavor]["invert_delta_phi"]
        
        cr_map = {
            "wplusjets": "wj",
            "top_tagger": "tt",
            "signal": "signal",
            "qcd_hadronic_closure": "closure",
            "test": "test"
        }

        self.OBJ_MAP = {
            "jet": "Jet",
            "bjet": "Jet",
            "cjet": "Jet",
            "lightjet": "Jet",
            "topjet": "FatJet",
            "wjet": "FatJet",
            "electron": "Electron",
            "tau": "Tau",
            "muon": "Muon"
        }

        self.cr_def = cr_map[self.processor]
        

    def systematic_variation_map(self, obj_type, obj_name, cutflow, var):
        """
        obj_type: lepton, AK4, AK8, met
        obj_name: tau, muon, electron, lightjet_JES/JER, bjet_JES/JER, jet_JES/JER, topjet_JES/JER, wjet_JES/JER, met
        cutflow: list of cuts to process
        var: systematic variation direction, e.g., "up"
        """

        
        # Useful for identifying cuts involving the top tagger
        cutflow_tmp = []
        cut_tmp = None

        corr_type = None

        for cut in cutflow:
            match = re.search(r"_([A-Za-z0-9]+)\)$", cut)
            if match:
                corr_type = match.group(1)
                obj_name = obj_name.split("_")[0]
            
            if "top_tagger" in cut:
                cut_tmp = cut
                break
            if match:
                cutflow_tmp.append(f"{cut}")
            else:
                cutflow_tmp.append(cut)
            


        if corr_type is None:
            # No systematic variation in this cutflow
            return

        # ==================================================
        # Creating the new selection criteria
        # ==================================================
        syst_var_masks = {}

        # dad
        suffix = f"_({obj_type}_{corr_type})" if self.cr is None else f"_{self.cr}_({obj_type}_{corr_type})"        
        
        if obj_name != "met" and corr_type is not None:
            varied_collection = getattr(
                self.objects["events"],
                self.OBJ_MAP[obj_name]
            )[self.variations[obj_name][corr_type][var]]

            objects_tmp = {
                **self.objects,
                f"{obj_name}s": varied_collection,
            }

            objects = cross_cleaning(objects_tmp, self.cc)

            syst_var_masks.update({
                f"{obj_name}_veto{suffix}": (ak.num(objects[f"{obj_name}s"]) == 0),
                f"one_{obj_name}{suffix}": (ak.num(objects[f"{obj_name}s"]) == 1),
                f"two_{obj_name}s{suffix}": (ak.num(objects[f"{obj_name}s"]) == 2),
                f"at_least_one_{obj_name}{suffix}": (ak.num(objects[f"{obj_name}s"]) >= 1),
                f"at_least_two_{obj_name}s{suffix}": (ak.num(objects[f"{obj_name}s"]) >= 2),
            })

        # Add met if there is a cut that starts with “met”
        if any(cut.startswith("met") for cut in cutflow):
            syst_var_masks[f"met_({obj_type}_{corr_type})"] = self.variations['met'][obj_type][corr_type][var]            


        # Add met if there is a cut that starts with "delta_phi"
        if any(cut.startswith("delta_phi_jet_met") for cut in cutflow):
            if self.invert_delta_phi:
                # CR B and CR D
                syst_var_masks[f"delta_phi_jet_met{suffix}"] = ~self.variations['delta_phi_jet_met'][obj_type][corr_type][var]
            else:
                syst_var_masks[f"delta_phi_jet_met{suffix}"] = self.variations['delta_phi_jet_met'][obj_type][corr_type][var]


        # =======================================================
        # Save the masks to be used
        # =======================================================
        selections_syst_var = PackedSelection(dtype='uint64')   
        for cut in cutflow:
            if cut in self.selections.names:
                selections_syst_var.add(cut, self.selections.all(cut))
            elif "top_tagger" in cut:
                # Remember that if events change, the top tagger must be reevaluated.
                continue
            else:
                selections_syst_var.add(f"{cut}", syst_var_masks[cut])


        # ---------------------------
        # Evaluate top tagger
        # ---------------------------  
        if cut_tmp is not None:
            region_mask_tmp = selections_syst_var.all(*cutflow_tmp)
    
            case_id = self.objects["events"].top_tagger_case_id
            mask_unresolved = (case_id == 1) | (case_id == 2) | (case_id == 9) | (case_id == 10)  
            mask_partially_resolved = (case_id == 3) | (case_id == 4) | (case_id == 11) | (case_id == 12)
            mask_resolved = (case_id == 5) | (case_id == 6) | (case_id == 7) | (case_id == 8) | (case_id == 13)

            objects = dict(self.objects)
            if obj_type in ["lepton", "met"]:
                # Top tagger events evaluated before (nominal case) 
                top_tagger_nom = (case_id >= 0)
            
            elif obj_type == "AK4":
                # Masks per case group
                top_tagger_nom = mask_resolved

                jets = self.objects["events"].Jet
                objects["bjets"] = jets[self.variations["bjet"][corr_type][var]]
                objects["lightjets"] = jets[self.variations["lightjet"][corr_type][var]]
                           
            elif obj_type == "AK8":
                top_tagger_nom = mask_unresolved

                fatjets = self.objects["events"].FatJet

                objects["topjets"] = fatjets[self.variations["topjet"][corr_type][var]]
                objects["wjets"] = fatjets[self.variations["wjet"][corr_type][var]]                
                
            else:
                raise ValueError(f"Unknown obj_type: {obj_type}")
    
            objects = cross_cleaning(objects, self.cc)
            region_mask_tmp = region_mask_tmp & ~top_tagger_nom        

    
            objects_top = get_topXfinder_masks(
                objects=objects,
                region_mask=region_mask_tmp,
                lepton_flavor=self.lepton_flavor,
                cross_cleaning=self.criteria["cross_cleaning"][self.lepton_flavor],
                top_tagger_cases=self.criteria["top_tagger"][self.lepton_flavor]["cases"],
                nworkers=self.criteria["top_tagger"][self.lepton_flavor]["nworkers"]
            )    

            # Restoring events
            top_mask = (objects_top["events"].top_tagger_case_id > 0) | top_tagger_nom
            top_mask_tmp = top_mask if "pass" in cut_tmp else ~top_mask
            selections_syst_var.add(cut_tmp, top_mask_tmp)

        
        # =======================================================
        #  Create cutflow table
        # =======================================================
        syst_name = f"{obj_type}_{corr_type}"
        fill_cutflow(cutflow, selections_syst_var, f"{self.table_name}_({map_object_level_var(syst_name)}_{var})", self.metadata, self.weights.weight())
        
        map_object_level_var

        # =============================================================
        #             Histograms
        # =============================================================
        hist = Histograms(self.lepton_flavor, self.processor, objects, self.weights.weight(), selections_syst_var, cutflow, is_syst_var =True)
        
        syst_direction = var.capitalize()
        self.histograms.update({
            f"{map_object_level_var(syst_name)}_{self.cr_def}_{self.lepton_flavor}_{self.year}{syst_direction}": 
            {
                "hist": hist.fill_histograms(),
                "count": 1,
                "sumw_all_weights": np.sum(self.weights.weight())
            }
        })        



        
    def prepare_cutflows_object_level(self):
        # metfilters is not affected by systematic variations.
        EXCLUDED_CUTS = {"metfilters"}
        
        objects_map = {
            # Leptons
            "electron": ("lepton", "SS"),
            "muon": ("lepton", "Rochester"),
            "tau": ("lepton", "TES"),
            # Met
            "met": ("met", "Uncluster"),
            # AK4 Jets
            "bjet": ("AK4", ["JES", "JER"]),
            #"cjet": ("AK4", ["JES", "JER"]),
            "lightjet": ("AK4", ["JES", "JER"]),
            # AK8 Jets
            "topjet": ("AK8", ["JES", "JER"]),
            "wjet": ("AK8", ["JES", "JER"]),
        }

        cut_map = {}

        for obj, (typ, systs) in objects_map.items():
            # obj: electron, muon, met, bjet, cjet, lightjet, top_jet, w_jet
            # typ: lepton, met, AK4, AK8
            # syst: SS, Rochester, TES, Uncluster, JES, JER       

            # Check availability of systematic variations for this object type using the MET recalculation
            tmp = self.variations.get('met', {}).get(typ)

            if not tmp:
                continue

            # Discarding objects without systematic variations
            if not isinstance(systs, list):
                systs = [systs]

            valid_systs = [s for s in systs if s in tmp]
            if not valid_systs:
                continue            
            


            if not isinstance(systs, list):
                # Convert to list, only affects electron, muon, tau, met
                systs = [systs]

            for syst in systs:
                new_cuts = []
                for cut in self.cut_names:
                    if cut in EXCLUDED_CUTS:
                        new_cuts.append(cut)
                    elif check_object_cut_dependency(obj, cut):
                        # See analysis_utils.py
                        new_cuts.append(f"{cut}_({typ}_{syst})")
                    else:
                        new_cuts.append(cut)
                
                if typ not in cut_map:
                    cut_map[typ] = {}
                    
                key_name = f"{obj}_{syst}" if len(systs) > 1 else obj
                cut_map[typ][key_name] = new_cuts  

        
        return cut_map
        
    def object_level(self):

        # ===========================================
        #  Cutflow for each object-level variation
        # ===========================================
        cutflow_maps = self.prepare_cutflows_object_level()
        
        for obj_type, obj_dict in cutflow_maps.items():
            for obj_name, cuts in obj_dict.items():
                self.systematic_variation_map(obj_type, obj_name, cuts, "up")
                self.systematic_variation_map(obj_type, obj_name, cuts, "down")
  
        return cutflow_maps        

    
    def event_level(self):
        for variation in self.weights.variations:
            fill_cutflow(self.cut_names, self.selections, f"cutflow_({variation})", self.metadata, self.weights.weight(variation))

            # =============================================================
            #             Histograms
            # =============================================================
            hist = Histograms(self.lepton_flavor, self.processor, self.objects, self.weights.weight(variation), self.selections, self.cut_names, is_syst_var =True)

            match = re.search(r'(\d{4})(Up|Down)$', variation)

            if match:
                year = match.group(1)  # "2017"
                direction = match.group(2)  # "Up" o "Down"
                
                # Crear el nombre base sin el año+dirección
                base_name = re.sub(rf'{year}{direction}$', '', variation)
                
                # Construir el nombre corregido
                corrected_name = f"{base_name}{self.cr_def}_{self.lepton_flavor}_{year}{direction}"

            else:
                corrected_name = variation
                
            self.histograms.update({
                corrected_name: {
                    "hist": hist.fill_histograms(),
                    "count": 1,
                    "sumw_all_weights": np.sum(self.weights.weight(variation))
                }
            })



      