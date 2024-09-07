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
    "met": ["JME", "met.json.gz"],
    "pujetid": ["JME", "jmar.json.gz"],
    "jetvetomaps": ["JME", "jetvetomaps.json.gz"]
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