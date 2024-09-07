import json
import hist 
import importlib.resources
import awkward as ak
from coffea import processor
from wprime_plus_b.processors.utils.analysis_utils import normalize

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
    def __init__(self, year="2017", yearmod="", tagger="deepJet", wp="M", output_type="hist"):
        self._year = year + yearmod
        self._tagger = tagger
        self._wp = wp
        self._output_type = output_type
        
        with importlib.resources.path("wprime_plus_b.data", "btagWPs.json") as path:
            with open(path, "r") as handle:
                btagWPs = json.load(handle)
        self._btagwp = btagWPs[self._tagger][self._year][self._wp]
        
        self.make_output = lambda: hist.Hist(
            hist.axis.StrCategory([], growth=True, name="dataset"),
            hist.axis.Variable([20, 30, 50, 70, 100, 140, 200, 300, 600, 1000], name="pt"),
            hist.axis.Regular(4, 0, 2.5, name="abseta"),
            hist.axis.IntCategory([0, 4, 5], name="flavor"),
            hist.axis.Regular(2, 0, 2, name="passWP"),
        )
        
    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        dataset = events.metadata["dataset"]
        
        phasespace_cuts = (
            (abs(events.Jet.eta) < 2.5)
            & (events.Jet.pt > 20.)
        )
        jets = events.Jet[phasespace_cuts]
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