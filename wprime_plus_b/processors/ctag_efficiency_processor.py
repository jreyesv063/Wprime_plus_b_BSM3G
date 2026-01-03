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
            working point {'L', 'M', 'T'}
    """

    def __init__(self, year="2017", yearmod="", tagger="deepJet", wp="T",
                 output_type="hist", run_systematics=False, qcd_data_driven=False, output_folder="",
                 debug=False):
        self._year = year
        self._tagger = tagger
        self._wp = wp
        self._output_type = output_type
        self.run_systematics = run_systematics

        # Load ctag working points
        with importlib.resources.path("wprime_plus_b.data", "ctagWPs.json") as path:
            with open(path, "r") as handle:
                ctag_working_points = json.load(handle)

        self._ctagwp = (
            ctag_working_points[tagger][year]["CvL_cut"][self._wp],
            ctag_working_points[tagger][year]["CvB_cut"][self._wp],
        )

        # Define hist output
        self.make_output = lambda: hist.Hist(
            hist.axis.StrCategory([], growth=True, name="dataset"),
            hist.axis.Variable([20, 30, 50, 70, 100, 140, 200, 300, 600, 1000],
                               name="pt", 
                               overflow=True),
            hist.axis.Regular(4, 0, 2.5, 
                              name="abseta", 
                              overflow=True),
            hist.axis.IntCategory([0, 4, 5],
                                  name="flavor"),
            hist.axis.StrCategory(["CvL", "CvB", "CvL_and_CvB"],
                                  name="passWP"),
        )

        # Precompute jet ID flags and PUID WPs
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

        # Build full mask for jets
        mask = (
            jet_veto_mask
            & (abs(events_masked.Jet.eta) < 2.5)
            & (events_masked.Jet.pt > 20.)
            & (events_masked.Jet.jetId == self.jet_id_flags[self._year]["tightLepVeto"])
            & (events_masked.Jet.puId == self.puid_wps["T"])
        )

        jets = events_masked.Jet[mask]

        print(" ==============================================")
        print(self._ctagwp[0], self._ctagwp[1])
        print(" ==============================================")
        # --- Apply c-tagging cuts ---
        passCvL = jets.btagDeepFlavCvL > self._ctagwp[0]
        passCvB = jets.btagDeepFlavCvB > self._ctagwp[1]
        passBoth = passCvL & passCvB  # pasa ambos

        out = {}

        # === HISTOGRAM OUTPUT ===

        print(f"[{dataset}] Número de jets después del mask:", ak.num(jets))
        print(f"[{dataset}] Jets que pasan CvL:", ak.sum(passCvL))
        print(f"[{dataset}] Jets que pasan CvB:", ak.sum(passCvB))
        print(f"[{dataset}] Jets que pasan ambos:", ak.sum(passBoth))

        if self._output_type == "hist":
            output = self.make_output()
            flat_jets = ak.flatten(jets)

            # Fill jets passing only CvL
            output.fill(
                dataset=dataset,
                pt=flat_jets.pt[ak.flatten(passCvL)],
                abseta=abs(flat_jets.eta[ak.flatten(passCvL)]),
                flavor=flat_jets.hadronFlavour[ak.flatten(passCvL)],
                passWP="CvL",
            )

            # Fill jets passing only CvB
            output.fill(
                dataset=dataset,
                pt=flat_jets.pt[ak.flatten(passCvB)],
                abseta=abs(flat_jets.eta[ak.flatten(passCvB)]),
                flavor=flat_jets.hadronFlavour[ak.flatten(passCvB)],
                passWP="CvB",
            )

            # Fill jets passing both CvL and CvB
            output.fill(
                dataset=dataset,
                pt=flat_jets.pt[ak.flatten(passBoth)],
                abseta=abs(flat_jets.eta[ak.flatten(passBoth)]),
                flavor=flat_jets.hadronFlavour[ak.flatten(passBoth)],
                passWP="CvL_and_CvB",
            )

            out["histograms"] = output

        # === ARRAY OUTPUT ===
        elif self._output_type == "array":
            flat_jets = ak.flatten(jets)

            features = {
                "pt": flat_jets.pt,
                "abseta": abs(flat_jets.eta),
                "flavor": flat_jets.hadronFlavour,
                "passCvL": ak.flatten(passCvL),
                "passCvB": ak.flatten(passCvB),
                "passBoth": ak.flatten(passBoth),
            }

            output = {
                name: processor.column_accumulator(normalize(array))
                for name, array in features.items()
            }
            out["arrays"] = output

        return {dataset: out}

    def postprocess(self, accumulator):
        return accumulator
