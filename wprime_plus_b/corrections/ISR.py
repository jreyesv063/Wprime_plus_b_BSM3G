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
    if dataset.startswith('WJetsToLNu') or dataset.startswith('DYJetsToLL'):
        
        if channel == "ll":
            json_correction = "wprime_plus_b/data/ISR_Zmumu_weight.json"
        elif channel == "ll+c":
            json_correction = "wprime_plus_b/data/ISR_Z+c_weight.json"

        # Determine the pdgId based on the dataset
        if dataset.startswith('WJetsToLNu'):
            pdgId = 24  # W boson
        else:
            pdgId = 23  # Z boson

        # get correction
        cset = correctionlib.CorrectionSet.from_file(json_correction)

        general_mask = (np.abs(events.GenPart.pdgId) == pdgId) & (events.GenPart.status == 62)
        ISR_Z_bosons = events.GenPart[general_mask]  # Select bosons from GenPart
        
        Z_pt = ak.firsts(ISR_Z_bosons.pt)  # Get the pt of the first boson in each event
        Z_pt = ak.fill_none(Z_pt,2000)
        Z_njets = ak.num(jets)
        

        # Calculate the ISR weight using the mother pt
        sf = cset[f"ISR_weight_{year}_UL"].evaluate("nominal", Z_njets, Z_pt)  # Calculate the ISR weight using the weight_ISR function
        
        
        # Apply variations if specified
        if variation == "nominal":
            # Calculate the ISR weight for the "up" variation
            sf_up = cset[f"ISR_weight_{year}_UL"].evaluate("nominal", Z_njets, Z_pt) 
            
            
            # Calculate the ISR weight for the "down" variation
            sf_down = cset[f"ISR_weight_{year}_UL"].evaluate("nominal", Z_njets, Z_pt) 
            
            # Add the calculated scale factors to the weights object
            weights.add(
                name="ISR_weight",  # Name of the weight
                weight=sf,  # Nominal weight
                weightUp=sf_up,  # Up variation weight
                weightDown=sf_down,  # Down variation weight
            )
        else:
            # If variation is not "nominal", add only the nominal scale factor
            weights.add(
                name="ISR_weight",  # Name of the weight
                weight=sf,  # Nominal weight
            )
