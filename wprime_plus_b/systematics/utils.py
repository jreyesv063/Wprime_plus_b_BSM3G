import numpy as np
import awkward as ak


def one_var_per_event_MET(met_threshold, delta_x_and_y_list, met):

    final_met_x = met.pt *  np.cos(met.phi) 
    final_met_y = met.pt * np.sin(met.phi) 


    new_met_list = {}

    for variation in delta_x_and_y_list:
       
        # Changes in X and Y axis
        delta_x = delta_x_and_y_list[variation]["delta_x"]["nom"]
        delta_y = delta_x_and_y_list[variation]["delta_y"]["nom"]


        # Remuevo el recalculo del MET (resto en lugar de sumar)
        # (new_met)_x  = (old_met)_x + (delta x)_x   ->  (new_met)_x  - (delta x)_x  = (old_met)_x 
        new_met_x = final_met_x - delta_x
        new_met_y = final_met_y - delta_y


        # Agrego la nueva variación up/down
        met_x_up = new_met_x + delta_x_and_y_list[variation]["delta_x"]["up"]
        met_x_down = new_met_x + delta_x_and_y_list[variation]["delta_x"]["down"]

        met_y_up = new_met_x + delta_x_and_y_list[variation]["delta_y"]["up"]
        met_y_down = new_met_x + delta_x_and_y_list[variation]["delta_y"]["down"]


        # Calculo el nuevo met up/down
        met_pt_up= np.sqrt(met_x_up**2 + met_y_up**2)
        met_pt_down= np.sqrt(met_x_down**2 + met_y_down**2)



        met_phi_up = np.arctan2(met_y_up, met_x_up)
        met_phi_down = np.arctan2(met_y_down, met_x_down)


        # Calculamos la mascara
        new_met_list[variation] = {
            "up": {
                "new_met_pt": met_pt_up,
                "new_met_phi": met_phi_up,
                "mask":  met_pt_up > met_threshold
            },
            "down": {
                "new_met_pt": met_pt_down,
                "new_met_phi": met_phi_down,
                "mask": met_pt_down > met_threshold
            },
        }


    return new_met_list
        



        
def update_region_map(region_map, map_variation, met_threshold):

    region_selection_map = {}
    
    met_value = met_threshold

    for variation in map_variation:
        if variation == "TES":
            for direction in ["up", "down"]:
                region_selection_map[f"{variation}_{direction}"] = [
                    f"one_tau_TES_{direction}" if x == "one_tau" else
                    f"met_{met_threshold}_TES_{direction}" if x == f"met_{met_value}" else x
                    for x in region_map
                ]
        
        elif variation == "ROCHESTER":
            for direction in ["up", "down"]:
                region_selection_map[f"{variation}_{direction}"] = [
                    f"met_{met_threshold}_ROCHESTER_{direction}" if x == f"met_{met_value}" else x
                    for x in region_map
                ]
        
        elif variation == "jet_JES":
            for direction in ["up", "down"]:
                region_selection_map[f"{variation}_{direction}"] = [
                    f"met_{met_threshold}_jet_JES_{direction}" if x == f"met_{met_value}" else x
                    for x in region_map
                ]
        
        elif variation == "jet_JER":
            for direction in ["up", "down"]:
                region_selection_map[f"{variation}_{direction}"] = [
                    f"met_{met_threshold}_jet_JER_{direction}" if x == f"met_{met_value}" else x
                    for x in region_map
                ]
        
        elif variation == "fatjet_JES":
            for direction in ["up", "down"]:
                region_selection_map[f"{variation}_{direction}"] = [
                    f"met_{met_threshold}_fatjet_JES_{direction}" if x == f"met_{met_value}" else x
                    for x in region_map
                ]
        
        elif variation == "fatjet_JER":
            for direction in ["up", "down"]:
                region_selection_map[f"{variation}_{direction}"] = [
                    f"met_{met_threshold}_fatjet_JER_{direction}" if x == f"met_{met_value}" else x
                    for x in region_map
                ]
        
        elif variation == "met_UNCLUSTERED":
            for direction in ["up", "down"]:
                region_selection_map[f"{variation}_{direction}"] = [
                    f"met_{met_threshold}_met_UNCLUSTERED_{direction}" if x == f"met_{met_value}" else x
                    for x in region_map
                ]


    return region_selection_map



def update_region_selection(map_variation, met_threshold, selections):
    
    for variation in map_variation:
        if variation == "TES":
            for direction in ["up", "down"]:
                selections.add(f"one_tau_TES_{direction}", map_variation[variation][direction]["one_tau"])
                selections.add(f"met_{met_threshold}_TES_{direction}", map_variation[variation][direction][f"met_{met_threshold}"])
                
        elif variation == "ROCHESTER":
            for direction in ["up", "down"]:
                selections.add(f"muon_veto_ROCHESTER_{direction}", map_variation[variation][direction]["muon_veto"])
                selections.add(f"met_{met_threshold}_ROCHESTER_{direction}", map_variation[variation][direction][f"met_{met_threshold}"])        

        
        elif variation == "fatjet_JES":
            for direction in ["up", "down"]:
                selections.add(f"met_{met_threshold}_fatjet_JES_{direction}", map_variation[variation][direction][f"met_{met_threshold}"])
    
                
        
        elif variation == "fatjet_JER":
            for direction in ["up", "down"]:
                selections.add(f"met_{met_threshold}_fatjet_JER_{direction}", map_variation[variation][direction][f"met_{met_threshold}"])
        

        elif variation == "jet_JES":
            for direction in ["up", "down"]:
                selections.add(f"met_{met_threshold}_jet_JES_{direction}", map_variation[variation][direction][f"met_{met_threshold}"])
                
        
        elif variation == "jet_JER":
            for direction in ["up", "down"]:
                selections.add(f"met_{met_threshold}_jet_JER_{direction}", map_variation[variation][direction][f"met_{met_threshold}"])
    
                
        elif variation == "met_UNCLUSTERED":
            for direction in ["up", "down"]:
                selections.add(f"met_{met_threshold}_met_UNCLUSTERED_{direction}", map_variation[variation][direction][f"met_{met_threshold}"])



