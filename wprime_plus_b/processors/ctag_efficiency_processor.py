import json
import hist 
import importlib.resources
import awkward as ak
from coffea import processor
from wprime_plus_b.processors.utils.analysis_utils import normalize
from wprime_plus_b.corrections.jetvetomaps import jetvetomaps_mask

class CTagEfficiencyProcessor(processor.ProcessorABC):
    """
    Compute ctag efficiencies for a tagger in a given working point

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

    def __init__(self, year="2017", yearmod="", tagger="deepJet", wp="T",
                 output_type="hist", run_systematics=False, output_folder="",
                 debug=False):
        self._year = year
        self._tagger = tagger
        self._wp = wp
        self.discriminator = "CvL_cut"
        self._output_type = output_type
        self.run_systematics = run_systematics
       

        # Load ctag working points
        with importlib.resources.path("wprime_plus_b.data", "ctagWPs.json") as path:
            with open(path, "r") as handle:
                ctag_working_points = json.load(handle)

        opposite_discriminator = "CvB_cut" if self.discriminator == "CvL_cut" else "CvL_cut"

        self._ctagwp = (
            ctag_working_points[tagger][year][self.discriminator][self._wp],
            ctag_working_points[tagger][year][opposite_discriminator][self._wp],
        )


        # Define hist output
        self.make_output = lambda: hist.Hist(
            hist.axis.StrCategory([], growth=True, name="dataset"),
            hist.axis.Variable([20, 30, 50, 70, 100, 140, 200, 300, 600, 1000],
                               name="pt", overflow=True),
            hist.axis.Regular(4, 0, 2.5, name="abseta", overflow=True),
            hist.axis.IntCategory([0, 4, 5], name="flavor"),
            hist.axis.Regular(2, 0, 2, name="passWP"),
        )

        # Precompute jet ID flags and PUID WPs (to avoid rebuilding every call)
        self.jet_id_flags = {
            "2016APV": {"loose": 1, "tight": 3, "tightLepVeto": 6},
            "2016":    {"loose": 1, "tight": 3, "tightLepVeto": 6},
            "2017":    {"tight": 2, "tightLepVeto": 6},
            "2018":    {"tight": 2, "tightLepVeto": 6},
        }

        self.puid_wps = {"fail": 0, "L": 4, "M": 6, "T": 7}

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        dataset = events.metadata["dataset"]

        # Good primary vertex requirement
        events_masked = events[events.PV.npvsGood > 0]

        # Jet veto mask
        jet_veto_mask = jetvetomaps_mask(events_masked.Jet, self._year, "jetvetomap")

        # Build full mask for jets in one step
        mask = (
            jet_veto_mask
            & (abs(events_masked.Jet.eta) < 2.5)
            & (events_masked.Jet.pt > 20.)
            & (events_masked.Jet.jetId == self.jet_id_flags[self._year]["tightLepVeto"])
            & (events_masked.Jet.puId == self.puid_wps["T"])
        )

        jets = events_masked.Jet[mask]

        # Apply c-tagging cuts
        passctag = (
            (jets.btagDeepFlavCvL > self._ctagwp[0])
            & (jets.btagDeepFlavCvB > self._ctagwp[1])
        )

        out = {}

        if self._output_type == "hist":
            output = self.make_output()
            flat_jets = ak.flatten(jets)
            flat_passctag = ak.flatten(passctag)

            output.fill(
                dataset=dataset,
                pt=flat_jets.pt,
                abseta=abs(flat_jets.eta),
                flavor=flat_jets.hadronFlavour,
                passWP=flat_passctag,
            )
            out["histograms"] = output

        elif self._output_type == "array":
            flat_jets = ak.flatten(jets)
            flat_passctag = ak.flatten(passctag)

            features = {
                "pt": flat_jets.pt,
                "abseta": abs(flat_jets.eta),
                "flavor": flat_jets.hadronFlavour,
                "pass_wp": flat_passctag,
            }

            output = {
                name: processor.column_accumulator(normalize(array))
                for name, array in features.items()
            }
            out["arrays"] = output

        return {dataset: out}
        


    def postprocess(self, accumulator):
        return accumulator
