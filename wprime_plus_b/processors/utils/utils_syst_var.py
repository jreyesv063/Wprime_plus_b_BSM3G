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
from wprime_plus_b.object_identification.top_selection import select_top_tagger


class Systematics:
    def __init__(
        self,
        cc,
        cr,
        year,  # New
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
        lepton_flavor,
        
    ):
        """
        Handles the propagation of systematic uncertainties at both object and event levels.
    
        This class orchestrates the variation of physics objects (Jets, Leptons, MET) 
        through 'Object-Level' systematics (e.g., JES, JER) and 'Event-Level' 
        systematics (e.g., pileup, trigger SFs, b-tagging weights).
    
        Key:
        --------------------
        1. Object-Level: Re-evaluates selection masks and high-level observables 
           (like Top Tagger cases or DeltaPhi) using varied pT/mass of particles.
        2. Event-Level: Applies varied weights to the nominal selection to estimate 
           uncertainties in normalization and shape.
        3. Integration: Automatically updates histograms and cutflow tables for 
           each systematic variation (Up/Down).
    
        Notes:
        -----
        The MET is used as a primary map for identifying which object variations 
        are available in the input dataset, except for AK8 jets which are handled 
        independently due to custom Top Tagger requirements.
        """

        self.cc = cc
        self.cr = cr
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
        self.year = year
        cr_map = {
            "ztoll": "zll",
            "wjets": "wjets",
            "wplusjets": "wj",
            "top_tagger": "tt",
            "signal": "signal",
            "qcd_hadronic_closure": "closure",
            "test": "test"
        }

        self.cr_def = cr_map[self.processor]
        
        
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #  Aux. function in object level variation: Obtain new masks
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def systematic_variation_map(self, obj_type, obj_name, cutflow, var):
        """
        obj_type: lepton, AK4, AK8, met
        obj_name: tau, muon, electron, lightjet_JES/JER, bjet_JES/JER, jet_JES/JER, topjet_JES/JER, wjet_JES/JER, met
        cutflow: list of cuts to process
        var: systematic variation direction, e.g., "up"
        """
        # =================================================
        # Identify corr_type and obj_type from cutflow
        # ==================================================
        corr_type = None

        for cut in cutflow:
            match = re.search(r"_([A-Za-z0-9]+)\)$", cut)

            if match:
                corr_type = match.group(1)
                obj_name = obj_name.split("_")[0]

        # Skip if no correction type is found, meaning this cutflow doesn't depend on the object variation
        if corr_type is None:
            return

        # ==================================================
        # Creating the new selection criteria
        # ==================================================
        syst_var_masks = {}

        # dad
        suffix = f"_({obj_type}_{corr_type})" if self.cr is None else f"_{self.cr}_({obj_type}_{corr_type})"        
        
        if obj_name != "met" and corr_type is not None:
            objects_tmp = copy(self.objects)
            obj_map = {
                "jet": "Jet",
                "bjet": "Jet",
                "lightjet": "Jet",
                "topjet": "FatJet",
                "wjet": "FatJet",
                "electron": "Electron",
                "tau": "Tau",
                "muon": "Muon"
            }
            
            objects_tmp[f"{obj_name}s"] = getattr(self.objects['events'], obj_map[obj_name])[self.variations[obj_name][corr_type][var]]
            objects = cross_cleaning(objects_tmp, self.cc)


            # Number of objects
            syst_var_masks.update({
                f"{obj_name}_veto{suffix}": (ak.num(objects[f"{obj_name}s"]) == 0),
                f"one_{obj_name}{suffix}": (ak.num(objects[f"{obj_name}s"]) == 1),
                f"two_{obj_name}s{suffix}": (ak.num(objects[f"{obj_name}s"]) == 2),
                f"at_least_one_{obj_name}{suffix}": (ak.num(objects[f"{obj_name}s"]) >= 1),
                f"at_least_two_{obj_name}s{suffix}": (ak.num(objects[f"{obj_name}s"]) >= 2),
            })
            

        else:
            objects = self.objects
            
           
        if any(cut.startswith("met") for cut in cutflow):
            #  Check if the object type (AK4, AK8, etc.) exists in the MET variations
            if obj_type in self.variations.get('met', {}):
                syst_var_masks[f"met_({obj_type}_{corr_type})"] = self.variations['met'][obj_type][corr_type][var]
        

        if any(cut.startswith("delta_phi_jet_met") for cut in cutflow):
            # Check if the object type (AK4, AK8, etc.) exists in the delta phi variations
            if obj_type in self.variations.get('delta_phi_jet_met', {}):
                syst_var_masks[f"delta_phi_jet_met{suffix}"] = self.variations['delta_phi_jet_met'][obj_type][corr_type][var]
                

        if any(cut.startswith("Z_boson") for cut in cutflow):
            # Check if the variation 'var' (up/down) exists for Z_boson
            if var in self.variations.get('Z_boson', {}):
                syst_var_masks[f"Z_boson{suffix}"] = self.variations['Z_boson'][var]
        
            
        if any(cut.startswith("leading_jet") for cut in cutflow) and "AK4" == obj_type:
            if obj_type in self.variations.get('leading_jet', {}):
                # General check: does this obj_type have a leading_jet variation?
                syst_var_masks[f"leading_jet{suffix}"] = self.variations['leading_jet'][obj_type][corr_type][var]
            

        # -----------------------------------------
        #  Top tagger
        # -----------------------------------------
        # Remember, top tagger uses AK4 and AK8 jets to trigger a top identification
        if obj_type in ["AK4", "AK8"]:
            if any(cut.startswith("top_tagger") for cut in cutflow):
                # corr_type: JES/JER
                if "AK4" == obj_type:
                    jets = self.objects["events"].Jet

                    # Create TLorentz vector with the variation
                    jets_var = ak.zip(
                        {
                            "pt": jets[f"{corr_type}_pt_{var}"],   
                            "eta": jets.eta,     
                            "phi": jets.phi,     
                            "mass":jets[f"{corr_type}_mass_{var}"] 
                        },
                        with_name="PtEtaPhiMLorentzVector"
                    )

                    objects["bjets"] = jets_var[self.variations["bjet"][corr_type][var]]
                    objects["lightjets"] = jets_var[self.variations["lightjet"][corr_type][var]]

    
                elif "AK8" == obj_type:    
                    fatjets = self.objects["events"].FatJet

                    # Create TLorentz vector with the variation
                    fatjets_var = ak.zip(
                        {
                            "pt": fatjets[f"{corr_type}_pt_{var}"],   
                            "eta": fatjets.eta,     
                            "phi": fatjets.phi,     
                            "mass": fatjets[f"{corr_type}_mass_{var}"] 
                        },
                        with_name="PtEtaPhiMLorentzVector"
                    )
                    
                    objects["topjets"] = fatjets_var[self.variations["topjet"][corr_type][var]]
                    objects["wjets"] = fatjets_var[self.variations["wjet"][corr_type][var]]                

    
                objects = cross_cleaning(objects, self.cc)

                objects, top_tagger_mask = select_top_tagger(
                    objects=objects,
                    region_mask=ak.ones_like(objects["events"].run, dtype=bool),  # All events to be evaluated
                    cross_cleaning=self.cc,
                    criteria=self.criteria["top_tagger"][self.lepton_flavor]
                )

                top_cut = next((c for c in cutflow if c.startswith("top_tagger")), None)

                if top_cut is None:
                    raise ValueError("Top tagger cut not found in cutflow")

                syst_var_masks[top_cut] = top_tagger_mask


        # =======================================================
        # Save the masks to be used
        # =======================================================
        selections_syst_var = PackedSelection(dtype='uint64')   
        for cut in cutflow:
            if cut in self.selections.names:
                selections_syst_var.add(cut, self.selections.all(cut))
            else:
                selections_syst_var.add(f"{cut}", syst_var_masks[cut])

        
        # =======================================================
        #  Create cutflow table
        # =======================================================
        syst_name = f"{obj_type}_{corr_type}"
        fill_cutflow(cutflow, selections_syst_var, f"{self.table_name}_({map_object_level_var(syst_name)}_{var})", self.metadata, self.weights.weight())

        
        # =============================================================
        #             Histograms
        # =============================================================
        hist = Histograms(self.lepton_flavor, self.processor, objects, self.weights.weight(), selections_syst_var, cutflow, is_syst_var = True)

        syst_direction = var.capitalize()
        self.histograms.update({
            f"{map_object_level_var(syst_name)}_{self.cr_def}_{self.lepton_flavor}_{self.year}{syst_direction}": {
                     "hist": hist.fill_histograms(),
                     "count": 1,
                     "sumw_all_weights": np.sum(self.weights.weight())
            }
        })        
        
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #  Aux. function in object level variation: Obtain cutflow
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def prepare_cutflows_object_level(self):
        """
        Builds a map of cutflows for each systematic variation.
        Optimized to avoid redundant AK4/AK8 processing while keeping 
        compatible naming conventions (obj_syst).
        """
        EXCLUDED_CUTS = {"metfilters"}
        
        objects_map = {
            "electron": ("lepton", "SS"),
            "muon": ("lepton", "Rochester"),
            "tau": ("lepton", "TES"),
            "met": ("met", "Uncluster"),
            "bjet": ("AK4", ["JES", "JER"]),
            "lightjet": ("AK4", ["JES", "JER"]),
            "topjet": ("AK8", ["JES", "JER"]),
            "wjet": ("AK8", ["JES", "JER"]),
        }

        cut_map = {}
        # We only want to process one representative object per type to avoid duplicates
        # For AK4, we pick 'bjet'. For AK8, we pick 'topjet'.
        REPRESENTATIVE_OBJECTS = {
            "lepton": ["electron", "muon", "tau"],
            "met": ["met"],
            "AK4": ["bjet"],      # lightjet will be skipped
            "AK8": ["topjet"]     # wjet will be skipped
        }

        for obj, (typ, systs) in objects_map.items():
            # SKIP: If the object is not the 'representative' for its type
            if obj not in REPRESENTATIVE_OBJECTS.get(typ, []):
                continue

            # Check variations in MET metadata
            tmp = self.variations.get('met', {}).get(typ)
                
            # Special case for AK8
            if not tmp and typ == "AK8":
                tmp = ["JES", "JER"]

            if not tmp:
                continue


            if not isinstance(systs, list):
                systs = [systs]
            
            valid_systs = [s for s in systs if s in tmp]
            if not valid_systs:
                continue

            for syst in systs:
                new_cuts = []
                for cut in self.cut_names:
                    # Top Tagger dependency check
                    is_top_dependency = cut.startswith("top_tagger") and typ in ["AK4", "AK8"]

                    # met dependecy check
                    is_met_cut = (cut == "met" or "delta_phi_jet_met" in cut)

                    if cut in EXCLUDED_CUTS:
                        new_cuts.append(cut)
       
                    elif is_top_dependency:
                        # Add top tagger dependency
                        new_cuts.append(f"{cut}_({typ}_{syst})")

                    elif is_met_cut:
                        # Add met dependency
                        if typ == "met" or typ in self.variations.get('met', {}):
                            new_cuts.append(f"{cut}_({typ}_{syst})")  
                        else:
                            new_cuts.append(cut) 

                    elif check_object_cut_dependency(obj, cut):
                        # Add object dependency
                        new_cuts.append(f"{cut}_({typ}_{syst})") 

                    else:
                        # Add nominal cut
                        new_cuts.append(cut)
                
                if typ not in cut_map:
                    cut_map[typ] = {}
                
                # Use the original naming convention (obj_syst) to avoid KeyErrors
                # but now it only runs once per type thanks to the skip at the top.
                key_name = f"{obj}_{syst}" if len(systs) > 1 else obj
                cut_map[typ][key_name] = new_cuts  

        return cut_map


    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #                  Systematic variations
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
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
            hist = Histograms(self.lepton_flavor, self.processor, self.objects, self.weights.weight(variation), self.selections, self.cut_names, is_syst_var = True)
            

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
      