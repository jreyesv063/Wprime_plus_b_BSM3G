# Import necessary libraries
import correctionlib  # Library for applying corrections and scale factors
import numpy as np  # Library for numerical operations
import awkward as ak  # Library for handling jagged arrays efficiently
from typing import Type, Tuple  # Import Type for type hints and Tuple for return types
from coffea.analysis_tools import Weights  # Import Weights class from Coffea for managing event weights

# Define the function to add ISR weights to the events
def ISR_weight(
    events,  # The collection of events to be weighted
    dataset,  # The name of the dataset being analyzed
    weights: Type[Weights],  # A Weights object from Coffea to which the ISR weights will be added
    year: str,  # The year of the dataset, used for year-specific corrections
    variation: str = "nominal",  # Specifies the variation of the weights to be applied. Default is "nominal"
) -> Tuple[ak.Array, ak.Array]:  # The function is expected to return a tuple of awkward Arrays, but it modifies the weights object in place
    """
    Adds ISR (Initial State Radiation) weights to the events based on the dataset, year, and variation.
    """

    # Apply ISR weights only to specific datasets
    if dataset.startswith('WJetsToLNu') or dataset.startswith('DYJetsToLL'):
        
        # Determine the pdgId based on the dataset
        if dataset.startswith('WJetsToLNu'):
            pdgId = 24  # W boson
        else:
            pdgId = 23  # Z boson
        
        # Create a mask for bosons (W or Z based on the dataset) with status 62
        general_mask = (np.abs(events.GenPart.pdgId) == pdgId) & (events.GenPart.status == 62)
        ISR_Z_bosons = events.GenPart[general_mask]  # Select bosons from GenPart
        
        Z_pt = ak.firsts(ISR_Z_bosons.pt)  # Get the pt of the first boson in each event

        # Create a mask for pt < 1000 GeV 
        pt_mask = (Z_pt < 1000)   # Mask for pt values < 1000 GeV

        in_pt_Z = Z_pt.mask[pt_mask]  # Apply the pt mask to the Z pt values
        
        mask_general = ak.fill_none(pt_mask, False)  # Create a general mask to identify if there is a boson in each sublist

        pt_Z = ak.fill_none(in_pt_Z, 1.0)  # Fill None values with 1.0 for any missing pt values
        
        # Calculate the ISR weight using the mother pt
        scale_factor = weight_ISR(pt_Z)  # Calculate the ISR weight using the weight_ISR function
        sf = ak.where(mask_general, scale_factor, 1)  # Apply scale factor if mask is true, else 1
        
        # Apply variations if specified
        if variation == "nominal":
            # Calculate the ISR weight for the "up" variation
            scale_factor_up = weight_ISR(pt_Z)  
            sf_up = ak.where(mask_general, scale_factor_up, 1)  # Apply "up" scale factor
            
            # Calculate the ISR weight for the "down" variation
            scale_factor_down = weight_ISR(pt_Z)
            sf_down = ak.where(mask_general, scale_factor_down, 1)  # Apply "down" scale factor
            
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
                weight=scale_factor,  # Nominal weight
            )

# Define the function to calculate the ISR weight
def weight_ISR(events_pt):  # Takes the transverse momentum (pt) of the events as input
    """
    Calculates the ISR weight based on the pt of the events.
    """
    # Polynomial parameters for the ISR weight calculation

    m_1 = -0.00010 # Slope of the linear function
    b_1 = 1.00316  # Intercept of the linear function for pt > 0
    b_0 = 0.99584  # Normalization factor
    
    # Calculate the ISR weight using the polynomial parameters
    weight = (events_pt * m_1 + b_1) / b_0  # Linear function to calculate weight
    
    return weight  # Return the calculated ISR weight
