import numpy as np
import awkward as ak

from coffea.analysis_tools import PackedSelection

from wprime_plus_b.processors.utils.analysis_utils import delta_r_mask

class SystematicVariation_objects:
    """
    Handles systematic variations for physics objects.
    Applies masks and cross-cleaning (delta-R) to filter objects.
    """

    def __init__(
        self,
        lepton_flavor: str = "tau",
        criteria: list = None,        
        object_collections: dict = None,  # e.g., {"muons": muons, "jets": jets, "electrons": electrons, ...}
        object_masks: dict = None,        # e.g., {"muons": muons_mask, "jets": jets_mask, ...}
        delta_r_threshold: float = 0.4,
        region_selection: list = None,
        delta_list_met: list = None,
        selections: PackedSelection = None
    ):

        self.lepton_flavor = lepton_flavor
        self.criteria = criteria
        self.events = object_collections["events"]
        
        self.object_collections = object_collections
        self.object_masks = object_masks
        self.delta_r_threshold = delta_r_threshold

        self.region_selection = region_selection

        self.delta_list_met = delta_list_met

        self.has_fatjets = np.sum(ak.num(self.events.FatJet)) > 0

        self.selections = selections

        
        # Object-specific systematic + ΔR veto configuration
        # Each entry: (collection, mask_dict, veto_against, is_fat, systematics_list)
        self.configs = {
            "muons":  (self.events.Muon , self.object_masks["muons"],  ["electrons"], False, ["Rochester"]),
            "taus":   (self.events.Tau, self.object_masks["taus"],   ["electrons","muons"], False, ["TES"]),
            "bjets":  (self.object_collections["jets_veto"], self.object_masks["bjets"],  ["electrons","muons","taus"], False, ["JES","JER"]),
            "jets":   (self.object_collections["jets_veto"], self.object_masks["jets"],   ["electrons","muons","taus","bjets"], False, ["JES","JER"]),
            "fatjets":(self.events.FatJet, self.object_masks["fatjets"],["electrons","muons","taus","bjets","jets"], True, ["JES","JER"]),
            "wjets":  (self.events.FatJet, self.object_masks["wjets"],  ["electrons","muons","taus","bjets","jets","fatjets"], True, ["JES","JER"]),
        }


    def filter_objects_with_systematics(self):
        """
        Applies systematic variation masks and ΔR cross-cleaning vetoes to each configured
        physics object collection.

        For each object type and each systematic group, it computes a **combined object-level mask**:

            `final_mask_up   = mask_up   AND all ΔR veto masks`
            `final_mask_down = mask_down AND all ΔR veto masks`

        ΔR vetoes are accumulated using a logical AND while preserving nested shape:

            `dr_mask = dr_mask AND delta_r_clean(...)`

        Returns
        -------
        dict
            Nested dictionary of filtered boolean masks for all systematic groups:

            {
                "<object>": {
                    "<systematic_group>": {
                        "up":   ak.Array mask,
                        "down": ak.Array mask
                    },
                    ...
                },
                ...
            }

        Notes
        -----
        - ΔR cleaning is **object-level**, not event-level.
        - If a veto reference collection is missing (`None`), it is treated as all-True mask
          (neutrally preserving the AND accumulation).
        - For fat-jet collections (`is_fat=True`), the ΔR threshold is doubled automatically.
        """
        new_object_results = {}

        def delta_r_clean(obj_array, reference_array, fat=False):
            """
            Computes ΔR veto mask between `obj_array` and a reference collection.
            Returns a nested boolean Awkward Array matching the shape of `obj_array`.
            If `reference_array` is None, returns an all-True mask of the same shape.
            """
            if reference_array is None:
                return True
            threshold = 2 * self.delta_r_threshold if fat else self.delta_r_threshold
            return delta_r_mask(obj_array, reference_array, threshold=threshold)

        # Loop over each object type
        for obj_key, (obj_array, mask_dict, veto_against, is_fat, systematics) in self.configs.items():
            new_object_results[obj_key] = {}

            for syst in systematics:
                # Select systematic up/down masks. Special cases use generic "up"/"down" keys,
                # otherwise construct keys as "JES/JER_up"/"JES/JER_down" for object-level masks.
                (mask_up, mask_down) = (mask_dict["up"], mask_dict["down"]) if syst in ("Rochester", "TES") else (mask_dict[f"{syst}_up"], mask_dict[f"{syst}_down"])
                
                # Initialize ΔR veto mask
                dr_mask = True

                # Accumulate AND across all veto reference collections
                for veto_obj_name in veto_against:
                    dr_mask = dr_mask & delta_r_clean(obj_array, self.object_collections.get(veto_obj_name), is_fat)

                # Store object-level masks
                new_object_results[obj_key][syst] = {"up": obj_array[mask_up & dr_mask], "down": obj_array[mask_down & dr_mask]}
                
          
        return new_object_results


    def recompute_met_systematics(self, met_criteria: list = None) -> dict:
        """
        Recomputes Missing Transverse Energy (MET) under systematic variations by:
          1. Converting nominal MET (pt, phi) → Cartesian components (x, y)
          2. Removing the nominal object-level correction contribution
          3. Applying up/down shifts for each systematic variation
          4. Recalculating MET pt and phi from shifted components
          5. Computing Δφ(jet, MET) cleaning masks per variation
          6. Storing MET values and all event-level selection masks
    
        Parameters
        ----------
        met_criteria : dict
            Dictionary containing event-level MET selection thresholds:
                {
                    "met_min" : float,
                    "delta_phi_jets_met" : float,
                    "invert_delta_phi" : bool
                }
    
        Returns
        -------
        dict
            Structure:
            {
                "<variation_name>" : {
                    "up" : {
                        "met_pt" : awkward.Array,
                        "met_phi" : awkward.Array,
                        "mask" : awkward.Array[bool],
                        "delta_phi_jet_met" : awkward.Array,
                        "delta_phi_mask" : awkward.Array[bool]
                    },
                    "down" : { same fields }
                },
                ...
            }
        """


        # Map systematic name → (object_key, variation_key)
        remap = {
            "Muon":           ("muons", "Rochester"),
            "Tau":            ("taus",  "TES"),
            "Jet_JES":        ("jets",  "JES"),
            "Jet_JER":        ("jets",  "JER"),
            "FatJet_JES":     ("fatjets","JES"),
            "FatJet_JER":     ("fatjets","JER"),
            'MET_uncluster':  ("met", "MET_uncluster"),
        }
    
        # ---------------------------------------------------------
        # Compute nominal MET x, y components (already corrected)
        # ---------------------------------------------------------
        final_met_x = self.object_collections["met"].pt  * np.cos(self.object_collections["met"].phi)
        final_met_y = self.object_collections["met"].pt  * np.sin(self.object_collections["met"].phi)
    
        # Container for recomputed MET values per systematic
        new_met_results = {}
    
        # ---------------------------------------------------------
        # Loop over available systematic MET corrections
        # ---------------------------------------------------------
        for var in self.delta_list_met:
    
            # ---------------------------------------------------------
            # Remove nominal correction contribution from MET x, y
            # ---------------------------------------------------------
            # Nominal corrections are subtracted because they were added
            # positively in the nominal MET definition in prior studies.
            new_met_x = final_met_x - self.delta_list_met[var]["delta_x"]["nom"]
            new_met_y = final_met_y - self.delta_list_met[var]["delta_y"]["nom"]
    
            # ---------------------------------------------------------
            # Apply up/down systematic shifts
            # ---------------------------------------------------------
            met_x_up   = new_met_x + self.delta_list_met[var]["delta_x"]["up"]
            met_x_down = new_met_x + self.delta_list_met[var]["delta_x"]["down"]
    
            met_y_up   = new_met_y + self.delta_list_met[var]["delta_y"]["up"]
            met_y_down = new_met_y + self.delta_list_met[var]["delta_y"]["down"]
    
            # ---------------------------------------------------------
            # Recalculate MET pt and phi from shifted x, y
            # ---------------------------------------------------------
            met_pt_up   = np.sqrt(met_x_up**2   + met_y_up**2)
            met_pt_down = np.sqrt(met_x_down**2 + met_y_down**2)
    
            met_phi_up   = np.arctan2(met_y_up,   met_x_up)
            met_phi_down = np.arctan2(met_y_down, met_x_down)
    
            # ---------------------------------------------------------
            # Compute MET pt selection mask
            # ---------------------------------------------------------
            # Keeps events above the minimum MET threshold
            met_mask_up   = met_pt_up   >= met_criteria["met_min"]
            met_mask_down = met_pt_down >= met_criteria["met_min"]
    
            # ---------------------------------------------------------
            # Compute Δφ(jet, MET) cleaning and selection mask
            # ---------------------------------------------------------
            # ak.zip wraps phi values into record structure so jets.delta_phi
            # can access them as object fields (e.g., .phi)
            delta_phi_up_jet   = self.object_collections["jets"].delta_phi(ak.zip({"phi": met_phi_up}))
            delta_phi_down_jet = self.object_collections["jets"].delta_phi(ak.zip({"phi": met_phi_down}))
    
            # ak.all ensures all jets in each event satisfy the Δφ cut
            good_delta_phi_up   = ak.all(np.abs(delta_phi_up_jet)   >= met_criteria["delta_phi_jets_met"], axis=-1)
            good_delta_phi_down = ak.all(np.abs(delta_phi_down_jet) >= met_criteria["delta_phi_jets_met"], axis=-1)
    
            # ---------------------------------------------------------
            # Optional inversion of Δφ mask
            # ---------------------------------------------------------
            if met_criteria["invert_delta_phi"]:
                good_delta_phi_up   = ~good_delta_phi_up
                good_delta_phi_down = ~good_delta_phi_down
    
            # ---------------------------------------------------------
            # Store results for this variation
            # ---------------------------------------------------------
            obj_key, syst_key = remap[var]

            # Only create the container if it does not already exist
            if obj_key not in new_met_results:
                new_met_results[obj_key] = {}   
                
            new_met_results[obj_key][syst_key] = {
                "up": {
                    "met_pt"   : met_pt_up,
                    "met_phi"  : met_phi_up,
                    "met_mask" : met_mask_up,
                    "delta_phi_jets_met" : delta_phi_up_jet,
                    "delta_phi_mask"    : good_delta_phi_up
                },
                "down": {
                    "met_pt"   : met_pt_down,
                    "met_phi"  : met_phi_down,
                    "met_mask" : met_mask_down,
                    "delta_phi_jets_met" : delta_phi_down_jet,
                    "delta_phi_mask"    : good_delta_phi_down
                }
            }
    
        return new_met_results

    
    def syst_var_mask(self, object_variations, met_map):

        syst_var_map = {
            "muon_Rochester": {
                "up": {
                    "muon_veto": (ak.num(object_variations['muons']['Rochester']['up']) == 0),
                    "one_muon": (ak.num(object_variations['muons']['Rochester']['up']) == 1),
                    "two_muons": (ak.num(object_variations['muons']['Rochester']['up']) == 2),
                    "at_least_two_muons": (ak.num(object_variations['muons']['Rochester']['up']) >= 2),
                    "met": met_map['muons']['Rochester']['up']['met_mask'],
                    "delta_phi_jet_met": met_map['muons']['Rochester']['up']['delta_phi_mask']
                    
                },
                "down": {
                    "muon_veto": (ak.num(object_variations['muons']['Rochester']['down']) == 0),
                    "one_muon": (ak.num(object_variations['muons']['Rochester']['down']) == 1),
                    "two_muons": (ak.num(object_variations['muons']['Rochester']['down']) == 2),
                    "at_least_two_muons": (ak.num(object_variations['muons']['Rochester']['down']) >= 2),
                    "met": met_map['muons']['Rochester']['down']['met_mask'],
                    "delta_phi_jet_met": met_map['muons']['Rochester']['down']['delta_phi_mask']
                }
            },
            "tau_TES": {
                "up": {
                    "tau_veto": (ak.num(object_variations['taus']['TES']['up']) == 0),                    
                    "one_tau": (ak.num(object_variations['taus']['TES']['up']) == 1),
                    "two_taus": (ak.num(object_variations['taus']['TES']['up']) == 2),
                    "met": met_map['taus']['TES']['up']['met_mask'],
                    "delta_phi_jet_met": met_map['taus']['TES']['up']['delta_phi_mask']
                },
                "down": {
                    "tau_veto": (ak.num(object_variations['taus']['TES']['down']) == 0),                    
                    "one_tau": (ak.num(object_variations['taus']['TES']['down']) == 1),
                    "two_taus": (ak.num(object_variations['taus']['TES']['down']) == 2),  
                    "met": met_map['taus']['TES']['down']['met_mask'],
                    "delta_phi_jet_met": met_map['taus']['TES']['down']['delta_phi_mask']
                }
            },
            "jet_JES": {
                "up": {
                    "bjet_veto": (ak.num(object_variations['jets']['JES']['up']) == 0),
                    "one_bjet": (ak.num(object_variations['jets']['JES']['up']) == 1),
                    "at_least_one_jet": (ak.num(object_variations['jets']['JES']['up']) >= 1),
                    "met": met_map['jets']['JES']['up']['met_mask'],
                    "delta_phi_jet_met": met_map['jets']['JES']['up']['delta_phi_mask']
                },
                "down": {
                    "bjet_veto": (ak.num(object_variations['jets']['JES']['down']) == 0),
                    "one_bjet": (ak.num(object_variations['jets']['JES']['down']) == 1),
                    "at_least_one_jet": (ak.num(object_variations['jets']['JES']['down']) >= 1),
                    "met": met_map['jets']['JES']['down']['met_mask'],
                    "delta_phi_jet_met": met_map['jets']['JES']['down']['delta_phi_mask']                    
                }
            },
            "jet_JER": {
                "up": {
                    "bjet_veto": (ak.num(object_variations['jets']['JER']['up']) == 0),
                    "one_bjet": (ak.num(object_variations['jets']['JER']['up']) == 1),
                    "at_least_one_jet": (ak.num(object_variations['jets']['JER']['up']) >= 1),
                    "met": met_map['jets']['JER']['up']['met_mask'],
                    "delta_phi_jet_met": met_map['jets']['JER']['up']['delta_phi_mask']                    
                },
                "down": {
                    "bjet_veto": (ak.num(object_variations['jets']['JER']['down']) == 0),
                    "one_bjet": (ak.num(object_variations['jets']['JER']['down']) == 1),
                    "at_least_one_jet": (ak.num(object_variations['jets']['JER']['down']) >= 1),
                    "met": met_map['jets']['JER']['down']['met_mask'],
                    "delta_phi_jet_met": met_map['jets']['JER']['down']['delta_phi_mask']                      
                }
            },
            "met_UNCLUSTERED": {
                "up": {
                    "met": met_map['met']['MET_uncluster']['up']['met_mask']
                },
                "down": {
                    "met": met_map['met']['MET_uncluster']['down']['met_mask']
                }                
            },            
           
        }

        # ------------------------------------------
        # Fatjets are not defined in all the samples
        # ------------------------------------------
        if self.has_fatjets:
            syst_var_map["fatjet_JES"] = {
                "up": {
                    "met": met_map['fatjets']['JES']['up']['met_mask'],
                    "delta_phi_jet_met": met_map['fatjets']['JES']['up']['delta_phi_mask']
                },
                "down": {
                    "met": met_map['fatjets']['JES']['down']['met_mask'],
                    "delta_phi_jet_met": met_map['fatjets']['JES']['down']['delta_phi_mask']                
                }
            }

            syst_var_map["fatjet_JER"] = {
                "up": {
                    "met": met_map['fatjets']['JER']['up']['met_mask'],
                    "delta_phi_jet_met": met_map['fatjets']['JER']['up']['delta_phi_mask']
                },
                "down": {
                    "met": met_map['fatjets']['JER']['down']['met_mask'],
                    "delta_phi_jet_met": met_map['fatjets']['JER']['down']['delta_phi_mask']                
                }
            }            

            
        return syst_var_map


    def object_var_syst(self, object_variations, met_map):

        object_syst_var_map = {
            "muon_Rochester": {
                "up": {
                    "new_muons": object_variations['muons']['Rochester']['up'],
                    "new_met_pt":  met_map['muons']['Rochester']['up']['met_pt'],
                    "new_met_phi":  met_map['muons']['Rochester']['up']['met_phi'],                    
                    "new_delta_phi_jets_met": met_map['muons']['Rochester']['up']['delta_phi_jets_met']
                },
                "down": {
                    "new_muons": object_variations['muons']['Rochester']['down'],
                    "new_met_pt":  met_map['muons']['Rochester']['down']['met_pt'],
                    "new_met_phi":  met_map['muons']['Rochester']['down']['met_phi'],                    
                    "new_delta_phi_jets_met": met_map['muons']['Rochester']['down']['delta_phi_jets_met']
                }
            },
            "tau_TES": {
                "up": {
                    "new_taus": object_variations['taus']['TES']['up'],
                    "new_met_pt":  met_map['taus']['TES']['up']['met_pt'],
                    "new_met_phi":  met_map['taus']['TES']['up']['met_phi'],                    
                    "new_delta_phi_jets_met": met_map['taus']['TES']['up']['delta_phi_jets_met']
                },
                "down": {
                    "new_taus": object_variations['taus']['TES']['down'],
                    "new_met_pt":  met_map['taus']['TES']['down']['met_pt'],
                    "new_met_phi":  met_map['taus']['TES']['down']['met_phi'],                    
                    "new_delta_phi_jets_met": met_map['taus']['TES']['down']['delta_phi_jets_met']
                }
            },
            "jet_JES": {
                "up": {
                    "new_bjets": object_variations['bjets']['JES']['up'],
                    "new_jets": object_variations['jets']['JES']['up'],
                    "new_met_pt":  met_map['jets']['JES']['up']['met_pt'],
                    "new_met_phi":  met_map['jets']['JES']['up']['met_phi'],                   
                    "new_delta_phi_jets_met": met_map['jets']['JES']['up']['delta_phi_jets_met']
                },
                "down": {
                    "new_bjets": object_variations['bjets']['JES']['down'],
                    "new_jets": object_variations['jets']['JES']['down'],
                    "new_met_pt":  met_map['jets']['JES']['down']['met_pt'],
                    "new_met_phi":  met_map['jets']['JES']['down']['met_phi'],                    
                    "new_delta_phi_jets_met": met_map['jets']['JES']['down']['delta_phi_jets_met']
                }
            },
            "jet_JER": {
                "up": {
                    "new_bjets": object_variations['bjets']['JER']['up'],
                    "new_jets": object_variations['jets']['JER']['up'],
                    "new_met_pt":  met_map['jets']['JER']['up']['met_pt'],
                    "new_met_phi":  met_map['jets']['JER']['up']['met_phi'],                   
                    "new_delta_phi_jets_met": met_map['jets']['JER']['up']['delta_phi_jets_met']
                },
                "down": {
                    "new_bjets": object_variations['bjets']['JER']['down'],
                    "new_jets": object_variations['jets']['JER']['down'],
                    "new_met_pt":  met_map['jets']['JER']['down']['met_pt'],
                    "new_met_phi":  met_map['jets']['JER']['down']['met_phi'],                   
                    "new_delta_phi_jets_met": met_map['jets']['JER']['down']['delta_phi_jets_met']
                }
            },
            "met_UNCLUSTERED": {
                "up": {
                    "new_met_pt":  met_map['met']['MET_uncluster']['up']['met_pt'],
                    "new_met_phi":  met_map['met']['MET_uncluster']['up']['met_phi'],                   
                    "new_delta_phi_jets_met": met_map['met']['MET_uncluster']['up']['delta_phi_jets_met']
                },
                "down": {
                    "new_met_pt":  met_map['met']['MET_uncluster']['down']['met_pt'],
                    "new_met_phi":  met_map['met']['MET_uncluster']['down']['met_phi'],                   
                    "new_delta_phi_jets_met": met_map['met']['MET_uncluster']['down']['delta_phi_jets_met']
                }                
            },            
           
        }

        # ------------------------------------------
        # Fatjets are not defined in all the samples
        # ------------------------------------------
        if self.has_fatjets:
            object_syst_var_map["fatjet_JES"] = {
                "up": {
                    "new_fatjets": object_variations['fatjets']['JES']['up'],
                    "new_wjets": object_variations['wjets']['JES']['up'],
                    "new_met_pt":  met_map['fatjets']['JES']['up']['met_pt'],
                    "new_met_phi":  met_map['fatjets']['JES']['up']['met_phi'],                   
                    "new_delta_phi_jets_met": met_map['fatjets']['JES']['up']['delta_phi_jets_met']
                },
                "down": {
                    "new_fatjets": object_variations['fatjets']['JES']['down'],
                    "new_wjets": object_variations['wjets']['JES']['down'],
                    "new_met_pt":  met_map['fatjets']['JES']['down']['met_pt'],
                    "new_met_phi":  met_map['fatjets']['JES']['down']['met_phi'],                   
                    "new_delta_phi_jets_met": met_map['fatjets']['JES']['down']['delta_phi_jets_met']
                }
            }

                
            object_syst_var_map["fatjet_JER"] = {
                "up": {
                    "new_fatjets": object_variations['fatjets']['JER']['up'],
                    "new_wjets": object_variations['wjets']['JER']['up'],
                    "new_met_pt":  met_map['fatjets']['JER']['up']['met_pt'],
                    "new_met_phi":  met_map['fatjets']['JER']['up']['met_phi'],                   
                    "new_delta_phi_jets_met": met_map['fatjets']['JER']['up']['delta_phi_jets_met']
                },
                "down": {
                    "new_fatjets": object_variations['fatjets']['JER']['down'],
                    "new_wjets": object_variations['wjets']['JER']['down'],
                    "new_met_pt":  met_map['fatjets']['JER']['down']['met_pt'],
                    "new_met_phi":  met_map['fatjets']['JER']['down']['met_phi'],                   
                    "new_delta_phi_jets_met": met_map['fatjets']['JER']['down']['delta_phi_jets_met']
                }
            }
            
        return object_syst_var_map

    def update_region_map(self,masks_map: list = None):

        # Output dictionary that will contain the new list of cut names for each syst + direction
        region_selection_map = {}

        # Loop over each systematic source in the mask map
        for syst, directions_map in masks_map.items():

            # For each systematic, produce both 'up' and 'down' variations
            for direction in ["up", "down"]:

                # List to store the new cut names for this systematic + direction
                new_cuts = []

                # Retrieve which cuts are actually affected for this direction
                # Example: directions_map["up"] → {"met": <bool array>, "muon_veto": <bool array>, ...}
                affected = directions_map.get(direction, {})

                # Loop over the base region cuts stored in the object
                for cut in self.region_selection:

                    # If this cut is affected by this systematic variation in this direction
                    if cut in affected:

                        # Build the new systematic-aware name
                        # e.g., "met_(met_UNCLUSTERED_up)"
                        new_name = f"{cut}_({syst}_{direction})"

                        # Append the renamed cut to the list for the region map
                        new_cuts.append(new_name)

                        # Register the real boolean mask into PackedSelection using the new name as key
                        mask = affected[cut]
                        self.selections.add(new_name, mask)

                    else:
                        # If the cut is not affected, keep it unchanged
                        new_cuts.append(cut)

                # Store the new cut list under the systematic_direction key
                # e.g., "tau_TES_up" → ["one_tau_(tau_TES_up)", "met_(tau_TES_up)", ...]
                region_selection_map[f"{syst}_{direction}"] = new_cuts

        # Return the full systematic-aware region cut map
        return region_selection_map
        
            
    def get_systematics_variation_mask(self):        

        new_objects = self.filter_objects_with_systematics()


        map_met_mask = self.recompute_met_systematics(met_criteria = self.criteria["met"][self.lepton_flavor])

        # -----------------------------------------
        #       Masks: Packed in self.selections
        # -----------------------------------------
        syst_var_masks = self.syst_var_mask(object_variations = new_objects, met_map = map_met_mask)
        region_selection_syst_var = self.update_region_map(masks_map = syst_var_masks)



        # ------------------------------------------------
        # New objects given the object-level corrections
        # ------------------------------------------------
        object_var_syst = self.object_var_syst(object_variations = new_objects, met_map = map_met_mask)

        
        return object_var_syst, region_selection_syst_var