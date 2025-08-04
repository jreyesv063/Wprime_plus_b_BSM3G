import json
import hist 
import awkward as ak
import importlib.resources
from coffea import processor
from wprime_plus_b.processors.utils.analysis_utils import normalize
from wprime_plus_b.corrections.jetvetomaps import jetvetomaps_mask


class BTagEfficiencyProcessor(processor.ProcessorABC):
    """
    Compute btag efficiencies for a tagger in a given working point

    Parameters:
    -----------
        year:
            year of the MC samples
        yearmod:
            year modifier {"", "APV"} (use "APV" for pre 2016 datasets)
        tagger:
            tagger name {'deepJet', 'deepCSV'}
        wp:
            worging point {'L', 'M', 'T'}
    """
    def __init__(self, year="2017", yearmod="", tagger="deepJet", wp="T", output_type="hist", run_systematics = False, output_folder = ""):
        self._year = year 
        self._tagger = tagger
        self._wp = wp
        self._output_type = output_type
        
        
        with importlib.resources.path("wprime_plus_b.data", "btagWPs.json") as path:
            with open(path, "r") as handle:
                btagWPs = json.load(handle)
        self._btagwp = btagWPs[self._tagger][self._year][self._wp]
        
        self.make_output = lambda: hist.Hist(
            hist.axis.StrCategory([], growth=True, name="dataset"),
            #hist.axis.Variable([20, 30, 50, 70, 100, 140, 200, 300, 600, 1000], name="pt"),
            #hist.axis.Regular(4, 0, 2.5, name="abseta"),
            hist.axis.Variable([20, 30, 50, 70, 100, 140, 200, 300, 600, 1000], name="pt", overflow=True), # Overflow is included.
            hist.axis.Regular(4, 0, 2.5, name="abseta", overflow=True),



            hist.axis.IntCategory([0, 4, 5], name="flavor"),
            hist.axis.Regular(2, 0, 2, name="passWP"),
        )

        print(self._wp, self._btagwp)
        
    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        dataset = events.metadata["dataset"]

        good_vertex_mask = (events.PV.npvsGood > 0) 

        events_masked =  ak.copy(events[good_vertex_mask])

        jet_veto_mask = jetvetomaps_mask(events_masked.Jet, self._year, "jetvetomap")
        jets_veto = events_masked.Jet[jet_veto_mask]

        jet_id_flags = {
            "2016APV": {
                "loose": 1,
                "tight": 3,
                "tightLepVeto": 6,
            },
            "2016": {
                "loose": 1,
                "tight": 3,
                "tightLepVeto": 6,
            },       
            "2017": {
                "tight": 2,
                "tightLepVeto": 6,
            },
            "2018": {
                "tight": 2,
                "tightLepVeto": 6,
            }
        }

        puid_wps = {
            "fail": 0,
            "L": 4,
            "M": 6,
            "T": 7,
        }
        
        phasespace_cuts = (
            (abs(jets_veto.eta) < 2.5)
            & (jets_veto.pt > 20.)
            & (jets_veto.jetId == jet_id_flags[self._year]["tightLepVeto"])
            & (jets_veto.puId == puid_wps["T"])
        )
        jets = jets_veto[phasespace_cuts]
        passbtag = jets.btagDeepFlavB > self._btagwp
        
        out = {}
        if self._output_type == "hist":
            output = self.make_output()
            output.fill(
                dataset=dataset,
                pt=ak.flatten(jets.pt),
                abseta=ak.flatten(abs(jets.eta)),
                flavor=ak.flatten(jets.hadronFlavour),
                passWP=ak.flatten(passbtag),
            )
            out["histograms"] = output
        
        elif self._output_type == "array":
            # select variables and put them in column accumulators
            features = {
                "pt": ak.flatten(jets.pt),
                "abseta": ak.flatten(abs(jets.eta)),
                "flavor": ak.flatten(jets.hadronFlavour),
                "pass_wp": ak.flatten(passbtag),
            }
            output = {
                feature_name: processor.column_accumulator(
                    normalize(feature_array)
                )
                for feature_name, feature_array in features.items()
            }
            out["arrays"] = output
        
        return {dataset: out}

    def postprocess(self, accumulator):
        return accumulator