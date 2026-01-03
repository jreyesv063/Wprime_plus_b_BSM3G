# Import necessary libraries
import correctionlib  # Library for applying corrections and scale factors
import numpy as np  # Library for numerical operations
import awkward as ak  # Library for handling jagged arrays efficiently
from typing import Type, Tuple  # Import Type for type hints and Tuple for return types
from coffea.analysis_tools import Weights  # Import Weights class from Coffea for managing event weights

# Define the function to add ISR weights to the events
def ISR_weight(
    events,  # The collection of events to be weighted
    jets,  # The collection of jets in the events
    dataset,  # The name of the dataset being analyzed
    weights: Type[Weights],  # A Weights object from Coffea to which the ISR weights will be added
    year: str,  # The year of the dataset, used for year-specific corrections
    channel: str,  # Specifies the channel of the dataset. Default is "mumu"
    variation: str = "nominal",  # Specifies the variation of the weights to be applied. Default is "nominal"
) -> Tuple[ak.Array, ak.Array]:  # The function is expected to return a tuple of awkward Arrays, but it modifies the weights object in place
    """
    Adds ISR (Initial State Radiation) weights to the events based on the dataset, year, and variation.
    """

    # Apply ISR weights only to specific datasets
    #if dataset.startswith('WJetsToLNu') or dataset.startswith('DYJetsToLL'):
    if dataset.startswith('DYJetsToLL'):
        # Determine the JSON file for ISR corrections based on the channel and year   
        if (
            dataset.startswith("WJetsToLNu") 
            or dataset.startswith("DYJetsToLL_M-50")
            or dataset.startswith("DYJetsToLL_M-10to50")            
        ):
            json_correction = f"wprime_plus_b/corrections/ISR/ISR_Zmumu_weight_MLM_{year}.json"
            ISR_type = "MLM"

        elif dataset.startswith("DYJetsToLL_nlo"):
            json_correction = f"wprime_plus_b/corrections/ISR/ISR_Zmumu_weight_FxFx_{year}.json"
            ISR_type = "FxFx"            

        else:
            raise ValueError(
                f"Dataset '{dataset}' is not one of the expected samples "
                "(WJetsToLNu*, DYJetsToLL_M-50*, DYJetsToLL_nlo*)."
            )



        # Determine the pdgId based on the dataset
        pdgId = 24 if dataset.startswith("WJetsToLNu") else 23

        # get correction
        cset = correctionlib.CorrectionSet.from_file(json_correction)

        general_mask = (
            (np.abs(events.GenPart.pdgId) == pdgId) 
            & (events.GenPart.status == 62)
        )

        boson_count = ak.sum(general_mask, axis=1)
        event_mask = (boson_count == 1)  # Create a mask for events with exactly one boson

        # Select the Z bosons from GenPart based on the general mask
        ISR_Z_bosons = ak.firsts(events.GenPart[general_mask])
        Z_pt = ak.fill_none(ISR_Z_bosons.pt, 1000) # Arbritrary number

        # Select jets where a Z boson pt is identified
        ISR_jets = jets.mask[event_mask]
        njets = ak.num(ISR_jets)

        # Variables for ISR weight calculation
        Z_pt_var = ak.fill_none(Z_pt, 0)
        njets_var = ak.fill_none(njets, 0)


        # Calculate the ISR weight using the mother pt
        sf = cset[f"ISR_weight_{year}_UL"].evaluate("nominal", njets_var, Z_pt_var)

        
        sf_nominal = ak.where(event_mask, sf, 1.0)  # Apply the ISR weight only where the mask is true, otherwise set to 1.0
       
        # Add nominal variation to the weights object
        weights.add(
            name=f"ISR_{ISR_type}_{year}",  # Name of the weight
            weight=sf_nominal,  # Nominal weight
        )
