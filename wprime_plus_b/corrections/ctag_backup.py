import json
import correctionlib
import numpy as np
import awkward as ak
import importlib.resources
from coffea import util
from typing import Type
from coffea.analysis_tools import Weights
from wprime_plus_b.corrections.utils import get_pog_json

class CTagCorrector:
    """
    CTag corrector class.

    Parameters:
    -----------
        sf_type:
            deepJet fixedWP c-tagging factors for UL 2017. 'up' and 'down' variations are available for the 
               different measurement types. The uncertainties are to be decorrelated between c jets ('wcharm'),
               b jets ('TnP') and light jets ('incl').
        working_point:
            working point {'L', 'M', 'T'}
        tagger:
            tagger {'deepJet', 'deepCSV'}
        year:
            dataset year {'2016', '2017', '2018'}
            
        year_mod:
            year modifier {"", "APV"}
        
        jets:
            Jet collection
        njets:
            Number of jets to use
        weights:
            Weights container from coffea.analysis_tools
        variation:
            if 'nominal' (default) add 'nominal', 'up' and 'down' variations to weights container. else, add only 'nominal' weights.
        full_run:
            False (default) if only one year is analized,
            True if the fullRunII data is analyzed.
            If False, the 'up' and 'down' systematics are be used.
            If True, 'up/down_correlated' and 'up/down_uncorrelated'
            systematics are used instead of the 'up/down' ones,
            which are supposed to be correlated/decorrelated
            between the different data years
    """
    def __init__(
        self,
        jets: ak.Array,
        weights: Type[Weights],
        working_point: str = "T",
        tagger: str = "deepJet",
        year: str = "2017",
        variation: str = "nominal",
        full_run: bool = False,
    ) -> None:

        self._year = year
        self._tagger = tagger
        self._jets = jets
        self._wp = working_point
        self._weights = weights
        self._variation = variation
        self._full_run = full_run
        self._discriminator = "CvB_cut"
        self._opposite_discriminator = "CvL_cut"
        
        with open(f"wprime_plus_b/corrections/efficiency_maps_ctag/ctag_eff_{self._tagger}_{self._wp}_{year}.coffea") as filename:
            self._efflookup = util.load(str(filename))
        
        # load ctagging working point (only for deepJet)
        # https://btv-wiki.docs.cern.ch/ScaleFactors/UL2017/
        with open("wprime_plus_b/json_files/ctagWPs.json", "r") as f:
            ctag_working_points = json.load(f)
        
        # define correction set
        self._cset = correctionlib.CorrectionSet.from_file(
            get_pog_json(json_name="ctag", year=year)
        )
                
        self._ctagwp = (
            ctag_working_points[tagger][year][self._discriminator][working_point],
            ctag_working_points[tagger][year][self._opposite_discriminator][working_point]
        ) 
        
  
        
    def add_ctag_weights(self, flavor: str) -> None:
            """
            Add c-tagging weights (nominal, up and down) to weights container for bc or light jets

            Parameters:
            -----------
                flavor:
                    hadron flavor {'b', 'c', 'light'}
            """
            # efficiencies
            eff = self.efficiency(flavor=flavor)
            
           
            # mask with events that pass the ctag working points
            passctag = self.passctag_mask(flavor=flavor)
    
            # nominal scale factors first method
            jets_sf = self.get_scale_factors(flavor=flavor, syst="central")

            ######### In case of applying the shape correction #####
            # nominal scale factors reshape
#             jets_sf_reshape = self.get_sf_reshape(flavor=flavor, syst="central")

#             jets_weight = ak.prod(jets_sf_reshape, axis=-1)
            ########################################################
            
            # nominal weights
            jets_weight = self.get_ctag_weight(eff, jets_sf, passctag) 
            
            if self._variation == "nominal":
            # systematics
                syst_up = "up_correlated" if self._full_run else "up"
                syst_down = "down_correlated" if self._full_run else "down"

                # up and down scale factors
                jets_sf_up = self.get_scale_factors(flavor=flavor, syst=syst_up)
                jets_sf_down = self.get_scale_factors(flavor=flavor, syst=syst_down)

                jets_weight_up = self.get_ctag_weight(eff, jets_sf_up, passctag)
                jets_weight_down = self.get_ctag_weight(eff, jets_sf_down, passctag)

                # add weights to Weights container
                self._weights.add(
                    name=f"{flavor}_jets_{self._wp}",
                    weight=jets_weight,
                    weightUp=jets_weight_up,
                    weightDown=jets_weight_down,
                )
            else:
                self._weights.add(
                    name=f"{flavor}_jets_{self._wp}",
                    weight=jets_weight,
                )
                
                
                
                
    def efficiency(self, flavor: str, fill_value=1) -> ak.Array:
        """compute the btagging efficiency for 'njets' jets"""
        return self._efflookup(
            self._jets.pt,
            np.abs(self._jets.eta),
            self._jets.hadronFlavour,
        )

    
    
    def passctag_mask(self, flavor, fill_value=True) -> ak.Array:
        """return the mask with jets that pass the c-tagging working point"""
        return ((self._jets["btagDeepFlavCvB"] > self._ctagwp[0]) & (self._jets["btagDeepFlavCvL"] > self._ctagwp[1])) 

    def get_scale_factors(self, flavor: str, syst="central", fill_value=1) -> ak.Array:
        """
        compute jets scale factors
        """
        return self.get_sf(flavor=flavor, syst=syst)
    
    def get_sf(self, flavor: str, syst: str = "central") -> ak.Array:
        """
        compute the scale factors for bc or light jets

        Parameters:
        -----------
            flavor:
                hadron flavor {'b', 'c' , 'light'}
            syst:
                Name of the systematic {'central', 'down', 'down_correlated', 'down_uncorrelated', 'up', 'up_correlated'}
        """    
        cset_key = "deepJet_wp"
        
        # hadron flavor definition: 5=b, 4=c, 0=udsg
        cset_methods ={
        "b" : ["TnP",5],
        "c" : ["wcharm",4],
        "light" : ["incl",0],
        }   

        
        # until correctionlib handles jagged data natively we have to flatten and unflatten
        j, nj = ak.flatten(self._jets), ak.num(self._jets)
        
        # In orden to filter according to the method we use use a dictionary
        method = cset_methods[flavor][0]
        flavor_number = cset_methods[flavor][1]
        
        #Masks

        # Corte de pT dependiendo del method. Considerando un corte mínimo y máximo
        if method == "TnP":
            jet_pt_mask = (j.pt >= 30.0) & (j.pt <= 200.0)
        elif method == "wcharm":
            jet_pt_mask = (j.pt >= 20.0) & (j.pt <= 210.0)
        elif method == "incl":
            jet_pt_mask = (j.pt >= 20.0) & (j.pt <= 1000.0)
        
        
        # eta mask
        jet_eta_mask = (np.abs(j.eta) < 2.499)
        
        # flavor mask
        jet_flavor_mask = (j.hadronFlavour == flavor_number)

        # CvL and CvB mask
        jet_CvL_mask = (j.btagDeepFlavCvL > self._ctagwp[1])
        jet_CvB_mask = (j.btagDeepFlavCvB > self._ctagwp[0])

        # jet mask
        in_jet_mask = (jet_eta_mask & jet_pt_mask & jet_flavor_mask & jet_CvL_mask & jet_CvB_mask)
        
        # jets that pass the flavor and eta masks
        in_jets = j.mask[in_jet_mask]
        
        # get jet transverse momentum, abs pseudorapidity and hadron flavour (replace None values with some 'in-limit' value)
        jets_pt = ak.fill_none(in_jets.pt, 0.0)
        jets_eta = ak.fill_none(np.abs(in_jets.eta), 0.0)
        jets_flavor = ak.fill_none(in_jets.hadronFlavour, flavor_number)
        
        sf = self._cset[cset_key].evaluate(
            syst,
            method,
            self._wp,
            np.array(jets_flavor),
            np.array(jets_eta),
            np.array(jets_pt),
        )
        sf = ak.where(in_jet_mask, sf, ak.ones_like(sf))
        
        #Imprimir los SF que se generan.
        return ak.unflatten(sf, nj)

    
    def get_sf_reshape(self, flavor: str, syst: str = "central") -> ak.Array:
        """
        Reshaping scale factors

        Parameters:
        -----------
            flavor:
                hadron flavor {'b', 'c' , 'light'}
            syst:
                Name of the systematic {'central', 'down', 'down_correlated', 'down_uncorrelated', 'up', 'up_correlated'}
        """
        
        cset_key = "deepJet_shape"
        
        
        # hadron flavor definition: 5=b, 4=c, 0=udsg
        cset_methods ={
        "b" : 5,
        "c" : 4,
        "light" : 0,
        }   

        
        # until correctionlib handles jagged data natively we have to flatten and unflatten
        j, nj = ak.flatten(self._jets), ak.num(self._jets)
        
        # In orden to filter according to the method we use use a dictionary
        flavor_number = cset_methods[flavor]
        
        #Mask
        
        # flavor mask
        jet_flavor_mask = (j.hadronFlavour == flavor_number)
        
        # jet mask
        in_jet_mask = jet_flavor_mask
        
        # jets that pass the flavor mask
        in_jets = j.mask[in_jet_mask]
        
        

        # get jet transverse momentum, abs pseudorapidity and hadron flavour (replace None values with some 'in-limit' value)
        jets_CvL = ak.fill_none(in_jets.btagDeepFlavCvL, 0.0)
        jets_CvB = ak.fill_none(np.abs(in_jets.btagDeepFlavCvB), 0.0)
        jets_flavor = ak.fill_none(in_jets.hadronFlavour, flavor_number)
        

        sf = self._cset[cset_key].evaluate(
            syst,
            np.array(jets_flavor),
            np.array(jets_CvL),
            np.array(jets_CvB),
        )
        sf = ak.where(in_jet_mask, sf, ak.ones_like(sf))

        return ak.unflatten(sf, nj)    

    @staticmethod
    def get_ctag_weight(eff: ak.Array, sf: ak.Array, passctag: ak.Array) -> ak.Array:
        """
        compute c-tagging weights

        see: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods

        Parameters:
        -----------
            eff:
                ctagging efficiencies
            sf:
                jets scale factors
            passctag:
                mask with jets that pass the c-tagging working point
        """
        

        # tagged SF = SF * eff / eff = SF
        tagged_sf = ak.prod(sf.mask[passctag], axis=-1)

        # untagged SF = (1 - SF * eff) / (1 - eff)
        untagged_sf = ak.prod(((1 - sf * eff) / (1 - eff)).mask[~passctag], axis=-1)

        return ak.fill_none(tagged_sf * untagged_sf, 1.0)
    
    



