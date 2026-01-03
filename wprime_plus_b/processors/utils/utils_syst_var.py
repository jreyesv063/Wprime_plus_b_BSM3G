import numpy as np
import awkward as ak
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

from coffea.analysis_tools import PackedSelection, Weights
from wprime_plus_b.processors.utils.utils_topXfinder import get_topXfinder_masks
from wprime_plus_b.systematics.syst_variations import SystematicVariation_objects
from wprime_plus_b.processors.utils.analysis_utils import fill_cutflow


# Histograms
from wprime_plus_b.processors.utils.histogram_utils import histograms_output_array



def run_top_tagger_task(task):
    """Wrapper del top-tagger para correr en threads (sin pickle)"""
    return get_topXfinder_masks(**task)


class Systematic_variation:

    def __init__(self,
        lepton_flavor: str = "tau",
        cut_names: list = None,
        criteria: dict = None,
        selections: PackedSelection = None,
        weights_container: Weights = None,
        metadata: Dict[str, Any] = None,
        objects: dict = None,
        object_variations: dict = None,
        delta_list: list = None,
        processor: str = "signal"
    ):
        self.lepton_flavor = lepton_flavor
        self.metadata = metadata
        self.objects = objects
        self.delta_list = delta_list
        self.cut_names = cut_names
        self.weights_container = weights_container
        self.selections = selections
        self.criteria = criteria
        self.object_variations = object_variations
        self.processor = processor

        self.variations_map = {
            "jet_JES":    ("jets", "bjets"),
            "jet_JER":    ("jets", "bjets"),
            "fatjet_JES": ("fatjets", "wjets"),
            "fatjet_JER": ("fatjets", "wjets"),
        }

    def syst_variation_mask(self, self_main,  table_name: str = "cutflow", nworkers: int = 4, name: str = ""):

        # Obtener masks de object-level systematics (igual que ya funciona)
        syst_var = SystematicVariation_objects(
            lepton_flavor = self.lepton_flavor,
            object_collections = self.objects,
            object_masks = self.object_variations,
            delta_r_threshold = self.criteria["cross_cleaning"][self.lepton_flavor],
            region_selection = self.cut_names,
            delta_list_met = self.delta_list,
            criteria = self.criteria,
            selections = self.selections,
        )

        objects_syst_var, region_selection_syst_var = syst_var.get_systematics_variation_mask()

        # Guardar si hay fatjets
        self.metadata["Are there Fatjets?"] = ak.sum(ak.num(self.objects["events"].FatJet)) > 0

        # Lista corregida de systematics up/down
        syst_var_object = [
            "muon_Rochester_up", "muon_Rochester_down",
            "tau_TES_up", "tau_TES_down",
            "jet_JES_up", "jet_JES_down",
            "jet_JER_up", "jet_JER_down",
            "fatjet_JES_up", "fatjet_JES_down",
            "fatjet_JER_up", "fatjet_JER_down",
        ]

        # Si no hay fatjets, quitar esas variaciones
        if not self.metadata["Are there Fatjets?"]:
            syst_var_object = [s for s in syst_var_object if "fatjet" not in s.lower()]

        # Preparar tasks del top-tagger
        top_tasks = []
        top_variations = []
        top_region_masks = []

        # Loop secuencial para construir region masks y llenar cutflow base por variation
        for variation, cuts in region_selection_syst_var.items():

            # 1) Inicializar cutflow para esa variation
            self.metadata[f"{table_name}_({variation})"] = {}
            self.metadata[f"{table_name}_({variation})_raw"] = {}

            # 2) Llenar cutflow parcial (nominal + raw lo maneja la función)
            fill_cutflow(
                metadata=self.metadata,
                cut_name="sumw",
                table_name=f"{table_name}_({variation})",
                weights=self.weights_container.weight()
            )

            selections_tmp = []
            mask_tmp = None

            if self.processor in ["wplusjets", "signal", "qcd_hadronic_closure"]:
                final_cut_str = f"fail_top_tagger_({variation})"
            elif self.processor == "top_tagger":
                final_cut_str = f"pass_top_tagger_({variation})"
            else:
                raise ValueError(
                    f"Processor '{self.processor}' is not supported when building the final top-tagger cut. "
                    f"Expected: 'wplusjets', 'signal', 'qcd_hadronic_closure', or 'top_tagger'."
                )

            for cut_name_tmp in cuts:
                selections_tmp.append(cut_name_tmp)
                mask_tmp = self.selections.all(*selections_tmp)
                fill_cutflow(
                    metadata=self.metadata,
                    cut_name=cut_name_tmp,
                    table_name=f"{table_name}_({variation})",
                    weights=self.weights_container.weight()[mask_tmp]
                )

            if mask_tmp is None:
                mask_tmp = ak.zeros_like(self.weights_container.weight(), dtype=bool)

            # Si no hay eventos, registrar fail y seguir (como antes)
            if ak.sum(mask_tmp) == 0:

                fill_cutflow(
                    metadata=self.metadata,
                    cut_name=final_cut_str,
                    table_name=f"{table_name}_({variation})",
                    weights=[]
                )
                continue

            # Copiar objetos y aplicar variaciones si corresponde
            objects_copy = {**self.objects}

            if any(variation.startswith(p) for p in ["jet", "fatjet"]):
                syst_var_name = variation.rsplit("_", 1)[0]
                if syst_var_name in self.variations_map:
                    direction = variation.rsplit("_", 1)[-1]
                    obj1, obj2 = self.variations_map[syst_var_name]
                    objects_copy[obj1] = objects_syst_var[syst_var_name][direction][f"new_{obj1}"]
                    objects_copy[obj2] = objects_syst_var[syst_var_name][direction][f"new_{obj2}"]

            # Empaquetar task para threads
            top_tasks.append({
                "lepton_flavor": self.lepton_flavor,
                "region_mask": mask_tmp,
                "objects": objects_copy,
                "top_tagger_cases": self.criteria["top_tagger"][self.lepton_flavor]["cases"],
                "cross_cleaning": self.criteria["cross_cleaning"][self.lepton_flavor],
                "invert_topXfinder": self.criteria["top_tagger"][self.lepton_flavor]["invert_top_tagger"]
            })

            top_variations.append(variation)
            top_region_masks.append(mask_tmp)

        # ============================================================
        #      Ejecutar TOP-TAGGER EN PARALELO (THREADS)
        # ============================================================

        if top_tasks:
            with ThreadPoolExecutor(max_workers=nworkers) as executor:
                top_results = list(executor.map(run_top_tagger_task, top_tasks))

            # Llenar cutflow final del top-tagger para cada variation (secuencial)
            for variation, top_result, region_mask in zip(top_variations, top_results, top_region_masks):
                mask_top_tmp, masks_tmp, njets_no_top_tmp, tops_tmp, selected_objects_tmp = top_result

                weights_var = self.weights_container.weight()[region_mask]
                final_weights = weights_var[mask_top_tmp]

                # 3) Guardar resultado en cutflow de esa variation
                fill_cutflow(
                    metadata=self.metadata,
                    cut_name=final_cut_str,
                    table_name=f"{table_name}_({variation})",
                    weights=final_weights,
                )

                histograms_output_array(
                    self_main = self_main,          
                    lepton_flavor =  self.lepton_flavor, 
                    njets_no_top = njets_no_top_tmp,     
                    tops = tops_tmp,                     
                    objects = selected_objects_tmp,      
                    mask = mask_top_tmp,                 
                    name = name,                         
                    syst_name = variation                
                )

        # ============================================================


    def get_syst_variation_event_level(self, name, region_mask, mask, self_main):
        print(f"syst_utils: {name}, {len(mask)}, {len(region_mask)}; {len(self.weights_container.weight())}")
        for variation_case, weights_case in self.weights_container._modifiers.items():
            self_main.add_feature(
                f"{variation_case}_{name}", weights_case[region_mask][mask]
            )

        for weight in self.weights_container.weightStatistics:
            filtered_weight = self.weights_container.partial_weight(include=[weight])[region_mask][mask]
            self_main.add_feature(weight, filtered_weight)


    def get_syst_variation_mask(self, self_main, table_name: str = "cutflow", nworkers: int = 4, name: str = ""):
        self.syst_variation_mask(self_main = self_main, table_name = table_name, nworkers = nworkers, name = name)
