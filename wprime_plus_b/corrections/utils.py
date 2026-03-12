import json
import gzip
import cloudpickle
import correctionlib
import numpy as np
import awkward as ak
import importlib.resources
from coffea import util
from typing import Type, Tuple
from coffea.lookup_tools import extractor
from coffea.analysis_tools import Weights
from coffea.nanoevents.methods.base import NanoEventsArray


# CorrectionLib files are available from
POG_CORRECTION_PATH = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration"

# summary of pog scale factors: https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/
POG_JSONS = {
    "muon": ["MUO", "muon_Z.json.gz"],
    "muon_highpt": ["MUO", "muon_HighPt.json.gz"],
    "electron": ["EGM", "electron.json.gz"],
    "tau": ["TAU", "tau.json.gz"],
    "pileup": ["LUM", "puWeights.json.gz"],
    "btag": ["BTV", "btagging.json.gz"],
    "ctag": ["BTV", "ctagging.json.gz"],
    "met": ["JME", "met.json.gz"],
    "pujetid": ["JME", "jmar.json.gz"],
    "jetvetomaps": ["JME", "jetvetomaps.json.gz"],
    "ak4_jec": ["JME", "jet_jerc.json.gz"],
    "ak8_jec": ["JME", "fatJet_jerc.json.gz"]
}

pog_years = {
    "2016": "2016postVFP_UL",
    "2016APV": "2016preVFP_UL",
    "2017": "2017_UL",
    "2018": "2018_UL",
}


def get_pog_json(json_name: str, year: str) -> str:
    """
    returns the path to the pog json file

    Parameters:
    -----------
        json_name:
            json name {'muon', 'electron', 'pileup', 'btag'}
        year:
            dataset year {'2016', '2017', '2018'}
    """
    if json_name in POG_JSONS:
        pog_json = POG_JSONS[json_name]
    else:
        print(f"No json for {json_name}")
    return f"{POG_CORRECTION_PATH}/POG/{pog_json[0]}/{pog_years[year]}/{pog_json[1]}"


def unflat_sf(sf: ak.Array, in_limit_mask: ak.Array, n: ak.Array):
    """
    get scale factors for in-limit objects (otherwise assign 1).
    Unflat array to original shape and multiply scale factors event-wise

    Parameters:
    -----------
        sf:
            Array with 1D scale factors
        in_limit_mask:
            Array mask for events with objects within correction limits
        n:
            Array with number of objects per event
    """
    sf = ak.where(in_limit_mask, sf, ak.ones_like(sf))
    return ak.fill_none(ak.prod(ak.unflatten(sf, n), axis=1), value=1)



def sample_crystal_ball(mean, sigma, alpha, n, size):
    """
    Sample random numbers from a Crystal Ball distribution using inverse CDF method.
    This is a vectorized implementation matching the C++ code in MuonScaRe.cc
    
    Parameters:
    - mean, sigma, alpha, n: CB parameters (arrays)
    - size: number of samples
    
    Returns:
    - array of random numbers following the CB distribution

    Ref: https://github.com/Vvvvvvvictor/HiggsZGammaAna/blob/010741748567a1986a12948c41bbed70d146c61a/HiggsDNA/higgs_dna/systematics/lepton_systematics.py#L1284
    """
    sqrt2 = numpy.sqrt(2.0)
    sqrtPiOver2 = numpy.sqrt(numpy.pi / 2.0)
    
    fa = numpy.abs(alpha)
    # Avoid division by zero: ensure fa > 0 and n > 1
    fa = numpy.maximum(fa, 1e-10)
    n = numpy.maximum(n, 1.0 + 1e-10)
    sigma = numpy.maximum(sigma, 1e-10)
    
    ex = numpy.exp(-fa * fa / 2.0)
    
    # CB normalization and helper terms
    with numpy.errstate(divide='ignore', invalid='ignore'):
        C1 = n / fa / (n - 1) * ex
        D1 = 2.0 * sqrtPiOver2 * _erf_vectorized(fa / sqrt2)
        
        B = n / fa - fa
        C = (D1 + 2 * C1) / C1
        D = (D1 + 2 * C1) / 2.0
        N = 1.0 / sigma / (D1 + 2 * C1)
        k = 1.0 / (n - 1)
        Ns = N * sigma
        NC = Ns * C1
        F = 1.0 - fa * fa / n
        G = sigma * n / fa
        
        # CDF at m-a*s and m+a*s
        cdfMa = NC / numpy.power(F + fa * sigma / G, n - 1)
        cdfPa = NC * (C - numpy.power(F + fa * sigma / G, 1 - n))
    
    # Handle NaN values in CDF calculations
    cdfMa = numpy.where(numpy.isnan(cdfMa), 0.0, cdfMa)
    cdfPa = numpy.where(numpy.isnan(cdfPa), 1.0, cdfPa)
    
    # Generate uniform random numbers
    u = numpy.random.uniform(0, 1, size)
    
    # Inverse CDF
    result = numpy.zeros(size)
    
    # Case 1: u < cdfMa (left tail)
    mask1 = u < cdfMa
    if numpy.any(mask1):
        with numpy.errstate(over='ignore', invalid='ignore'):
            result[mask1] = mean[mask1] + G[mask1] * (F[mask1] - numpy.power(NC[mask1] / u[mask1], k[mask1]))
    
    # Case 2: u > cdfPa (right tail)
    mask2 = u > cdfPa
    if numpy.any(mask2):
        with numpy.errstate(over='ignore', invalid='ignore'):
            result[mask2] = mean[mask2] - G[mask2] * (F[mask2] - numpy.power(C[mask2] - u[mask2] / NC[mask2], -k[mask2]))
    
    # Case 3: cdfMa <= u <= cdfPa (Gaussian core)
    mask3 = ~mask1 & ~mask2
    if numpy.any(mask3):
        result[mask3] = mean[mask3] - sqrt2 * sigma[mask3] * _erfinv_vectorized(
            (D[mask3] - u[mask3] / Ns[mask3]) / sqrtPiOver2
        )
    
    # Handle NaN or inf values in result
    result = numpy.where(numpy.isnan(result) | numpy.isinf(result), mean, result)
    
    return result


def get_era(year, run, era_map):
    for entry in era_map[year]["eras"]:
        if entry["run_min"] <= run <= entry["run_max"]:
            return entry["era"]
    return None
    