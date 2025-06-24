import numpy as np
import awkward as ak

from coffea.analysis_tools import PackedSelection



def calc_mt(pt_lep, pt_met, dphi, epsilon=1e-6):
    return np.sqrt(2 * np.maximum(pt_lep, epsilon) * pt_met * (1 - np.cos(dphi)))


def delta_phi_numpy(phi1, phi2):
    dphi = phi1 - phi2
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    return np.abs(dphi)



def one_var_per_event_MET(
    jets,
    met_threshold,
    delta_x_and_y_list,
    met,
    lepton=None,
    mt_threshold=None,
    mt_invert=False,
    delta_phi_cut=None,
    invert_delta_phi_cut = False
):
    final_met_x = met.pt * np.cos(met.phi)
    final_met_y = met.pt * np.sin(met.phi)

    new_met_list = {}

    for variation in delta_x_and_y_list:
        delta_x = delta_x_and_y_list[variation]["delta_x"]["nom"]
        delta_y = delta_x_and_y_list[variation]["delta_y"]["nom"]

        new_met_x = final_met_x - delta_x
        new_met_y = final_met_y - delta_y

        met_x_up = new_met_x + delta_x_and_y_list[variation]["delta_x"]["up"]
        met_x_down = new_met_x + delta_x_and_y_list[variation]["delta_x"]["down"]

        met_y_up = new_met_y + delta_x_and_y_list[variation]["delta_y"]["up"]
        met_y_down = new_met_y + delta_x_and_y_list[variation]["delta_y"]["down"]

        met_pt_up = np.sqrt(met_x_up**2 + met_y_up**2)
        met_pt_down = np.sqrt(met_x_down**2 + met_y_down**2)

        met_phi_up = np.arctan2(met_y_up, met_x_up)
        met_phi_down = np.arctan2(met_y_down, met_x_down)

        # Cálculo opcional de masa transversa
        if lepton is not None:
            if variation == "Tau":
                l_up = ak.firsts(lepton["up"])
                l_down = ak.firsts(lepton["down"])
            else:
                l_up = ak.firsts(lepton["nom"])
                l_down = l_up

            delta_phi_up = delta_phi_numpy(l_up.phi, met_phi_up)
            delta_phi_down = delta_phi_numpy(l_down.phi, met_phi_down)

            mt_up = calc_mt(l_up.pt, met_pt_up, delta_phi_up)
            mt_down = calc_mt(l_down.pt, met_pt_down, delta_phi_down)
            
        else:
            mt_up = None
            mt_down = None

        entry = {
            "up": {
                "new_met_pt": met_pt_up,
                "new_met_phi": met_phi_up,
                "mt": mt_up,
                "mask": met_pt_up > met_threshold
            },
            "down": {
                "new_met_pt": met_pt_down,
                "new_met_phi": met_phi_down,
                "mt": mt_down,
                "mask": met_pt_down > met_threshold
            },
        }

        if delta_phi_cut is not None:
            # Delta_phi_met_jets
            met_up_dummy = ak.zip({"phi": met_phi_up})
            met_down_dummy = ak.zip({"phi": met_phi_down})
    
    
            delta_phi_up_jet = jets.delta_phi(met_up_dummy)
            delta_phi_down_jet = jets.delta_phi(met_down_dummy)
    
            good_delta_phi_up = ak.all(np.abs(delta_phi_up_jet) >= delta_phi_cut, axis=-1)
            good_delta_phi_down = ak.all(np.abs(delta_phi_down_jet) >= delta_phi_cut, axis=-1)
    
            
            # Invert the mask if necessary: less than the mt_cut
            if invert_delta_phi_cut:
                good_delta_phi_up = ~good_delta_phi_up
                good_delta_phi_down = ~good_delta_phi_down
    
            entry["up"]["delta_phi"] = good_delta_phi_up
            entry["down"]["delta_phi"] = good_delta_phi_down
            

        if mt_threshold is not None and mt_up is not None and mt_down is not None:
            entry["up"]["mt_mask"] = mt_up < mt_threshold if mt_invert else mt_up > mt_threshold
            entry["down"]["mt_mask"] = mt_down < mt_threshold if mt_invert else mt_down > mt_threshold

        new_met_list[variation] = entry

    return new_met_list



def update_region_map(region_map, map_variation, met_threshold, mt_threshold=None, delta_phi = None):
    region_selection_map = {}

    for variation in map_variation:
        for direction in ["up", "down"]:
            new_region = []

            for x in region_map:
                # TES (tau-related)
                if variation == "TES":
                    if "tau" in x and not x.startswith("trigger"):
                        new_region.append(f"{x}_{variation}_{direction}")
                    elif x == f"met_{met_threshold}":
                        new_region.append(f"met_{met_threshold}_TES_{direction}")
                    #elif mt_threshold is not None and x.startswith(f"mt_{mt_threshold}"):
                    elif mt_threshold is not None and x == f"mt_{mt_threshold}":
                        new_region.append(f"{x}_{variation}_{direction}")
                    elif delta_phi is not None and x.startswith("delta_phi_jet_met_"):
                        new_region.append(f"{x}_{variation}_{direction}")
                    else:
                        new_region.append(x)

                # ROCHESTER (muon-related)
                elif variation == "ROCHESTER":
                    if "muon" in x and not x.startswith("trigger"):
                        new_region.append(f"{x}_{variation}_{direction}")
                    elif x == f"met_{met_threshold}":
                        new_region.append(f"met_{met_threshold}_ROCHESTER_{direction}")
                    #elif mt_threshold is not None and x.startswith(f"mt_{mt_threshold}"):
                    elif mt_threshold is not None and x == f"mt_{mt_threshold}":
                        new_region.append(f"{x}_{variation}_{direction}")
                    elif delta_phi is not None and x.startswith("delta_phi_jet_met_"):
                        new_region.append(f"{x}_{variation}_{direction}")
                    else:
                        new_region.append(x)

                # jet JES
                elif variation == "jet_JES":
                    if "bjet" in x or "jet" in x:
                        new_region.append(f"{x}_{variation}_{direction}")
                    elif x == f"met_{met_threshold}":
                        new_region.append(f"met_{met_threshold}_jet_JES_{direction}")
                    #elif mt_threshold is not None and x.startswith(f"mt_{mt_threshold}"):
                    elif mt_threshold is not None and x == f"mt_{mt_threshold}":
                        new_region.append(f"{x}_{variation}_{direction}")
                    elif delta_phi is not None and x.startswith("delta_phi_jet_met_"):
                        new_region.append(f"{x}_{variation}_{direction}")
                    else:
                        new_region.append(x)

                # jet JER
                elif variation == "jet_JER":
                    if "bjet" in x or "jet" in x:
                        new_region.append(f"{x}_{variation}_{direction}")
                    elif x == f"met_{met_threshold}":
                        new_region.append(f"met_{met_threshold}_jet_JER_{direction}")
                    elif mt_threshold is not None and x == f"mt_{mt_threshold}":
                        new_region.append(f"{x}_{variation}_{direction}")
                    elif delta_phi is not None and x.startswith("delta_phi_jet_met_"):
                        new_region.append(f"{x}_{variation}_{direction}")
                    else:
                        new_region.append(x)

                # fatjet JES
                elif variation == "fatjet_JES":
                    if x == f"met_{met_threshold}":
                        new_region.append(f"{x}_{variation}_{direction}")
                    elif mt_threshold is not None and x == f"mt_{mt_threshold}":
                        new_region.append(f"{x}_{variation}_{direction}")
                    elif delta_phi is not None and x.startswith("delta_phi_jet_met_"):
                        new_region.append(f"{x}_{variation}_{direction}")
                    else:
                        new_region.append(x)

                # fatjet JER
                elif variation == "fatjet_JER":
                    if x == f"met_{met_threshold}":
                        new_region.append(f"{x}_{variation}_{direction}")
                    elif mt_threshold is not None and x == f"mt_{mt_threshold}":
                        new_region.append(f"{x}_{variation}_{direction}")
                    elif delta_phi is not None and x.startswith("delta_phi_jet_met_"):
                        new_region.append(f"{x}_{variation}_{direction}")
                    else:
                        new_region.append(x)

                # met UNCLUSTERED
                elif variation == "met_UNCLUSTERED":
                    if x == f"met_{met_threshold}":
                        new_region.append(f"{x}_{variation}_{direction}")
                    elif mt_threshold is not None and x == f"mt_{mt_threshold}":
                        new_region.append(f"{x}_{variation}_{direction}")
                    elif delta_phi is not None and x.startswith("delta_phi_jet_met_"):
                        new_region.append(f"{x}_{variation}_{direction}")
                    else:
                        new_region.append(x)

            # Agregar al mapa final
            region_selection_map[f"{variation}_{direction}"] = new_region

    return region_selection_map




def is_flat_boolean_array(array):
    # Debe ser una instancia de Array o ndarray
    if not isinstance(array, (ak.Array, np.ndarray)):
        return False

    try:
        # Intenta convertir a numpy para verificar que sea booleano plano
        flat = ak.to_numpy(array)
        return flat.dtype == bool
    except Exception:
        return False


def update_region_selection(map_variation, selections_nominal):
    """
    Crea un PackedSelection por variación y dirección.
    Copia las máscaras de selections_nominal, pero reemplaza aquellas que se modifican en la variación.
    """
    selection_map = {}

    # Máscaras nominales base
    base_names = selections_nominal.names
    base_masks = {name: selections_nominal.all(name) for name in base_names}

    for variation in map_variation:
        for direction in ["up", "down"]:
            sel = PackedSelection(dtype="uint64")

            # Máscaras modificadas por la variación
            mod_keys = set()
            for key, mask in map_variation[variation][direction].items():
                if is_flat_boolean_array(mask):
                    name = f"{key}_{variation}_{direction}"
                    sel.add(name, mask)
                    mod_keys.add(key)  # Guardamos para saber cuáles no copiar del nominal

            # Agregar el resto de máscaras nominales no modificadas
            for name, mask in base_masks.items():
                key_base = name.split("_")[0]  # por ejemplo, "met_180" → "met"
                if key_base not in mod_keys:
                    sel.add(name, mask)

            # Guardar esta selección
            selection_map[f"{variation}_{direction}"] = sel

    return selection_map
                
                
            
            


