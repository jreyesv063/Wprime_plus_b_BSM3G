import yaml
import awkward as ak
import numpy as np


class Histograms:
    def __init__(
        self,
        lepton,
        processor,
        objects,
        weights,
        selections,
        cutflow_names,
        is_syst_var: bool = False,
    ):
        self.objects = objects
        self.weights = weights
        self.selections = selections
        self.processor = processor
        self.lepton = lepton

        self.is_syst_var = is_syst_var

        if self.is_syst_var:
            self.cuts_to_store = [cutflow_names[-1]]
        else:
            self.cuts_to_store = cutflow_names

        self.cutflow_names = cutflow_names

        # Store only NumPy arrays
        self.histograms = {}

        self.get_histogram_config()
        self.create_histograms()

    # ---------------------------------------------------------
    # Read YAML config and keep only plottable variables
    # ---------------------------------------------------------
    def get_histogram_config(self):
        with open(
            f"wprime_plus_b/selection_criteria/{self.processor}/Histograms_{self.lepton}.yaml",
            "r",
        ) as file:
            config = yaml.safe_load(file)

        self.histogram_list = {
            cat: {
                var: spec
                for var, spec in vars.items()
                if spec.get("include_plot")
            }
            for cat, vars in config.items()
        }

        # Remove empty categories
        self.histogram_list = {
            cat: vars for cat, vars in self.histogram_list.items() if vars
        }

    # ---------------------------------------------------------
    # Initialize lightweight histogram structures
    # ---------------------------------------------------------
    def create_histograms(self):
        for category, variables in self.histogram_list.items():
            for var_name, info in variables.items():

                # Define bin edges
                if info["type"] == "Regular":
                    edges = np.linspace(
                        info["start"],
                        info["stop"],
                        info["bins"] + 1,
                        dtype=np.float64,
                    )
                else:
                    edges = np.asarray(
                        info["edges"],
                        dtype=np.float64,
                    )

                # One histogram per variable and cut
                self.histograms[var_name] = {
                    cut: {
                        "edges": edges.copy(),
                        "sumw": np.zeros(
                            len(edges) - 1,
                            dtype=np.float64,
                        ),
                        "counts": np.zeros(
                            len(edges) - 1,
                            dtype=np.int64,
                        ),
                    }
                    for cut in self.cuts_to_store
                }

    # ---------------------------------------------------------
    # Fill histograms
    # ---------------------------------------------------------
    def fill_histograms(self):

        def fill_with_overflow(data, weights, edges):

            # Weighted histogram
            sumw, _ = np.histogram(
                data,
                bins=edges,
                weights=weights,
            )

            # Unweighted event counts per bin
            counts, _ = np.histogram(
                data,
                bins=edges,
            )

            # Underflow goes into first bin
            under_mask = data < edges[0]
            if np.any(under_mask):
                sumw[0] += np.sum(weights[under_mask])
                counts[0] += np.sum(under_mask)

            # Overflow goes into last bin
            over_mask = data >= edges[-1]
            if np.any(over_mask):
                sumw[-1] += np.sum(weights[over_mask])
                counts[-1] += np.sum(over_mask)

            return sumw, counts

        # -----------------------------------------------------
        # Systematics: fill only final cut
        # -----------------------------------------------------
        if self.is_syst_var:

            cut_name = self.cuts_to_store[0]
            mask = self.selections.all(*self.cutflow_names)
            weights = self.weights[mask]

            for category, variables in self.histogram_list.items():
                for var_name, info in variables.items():

                    data = eval(
                        info["expression"],
                        {"objects": self.objects, "ak": ak, "np": np},
                    )[mask]

                    h = self.histograms[var_name][cut_name]

                    sumw, counts = fill_with_overflow(
                        data,
                        weights,
                        h["edges"],
                    )

                    h["sumw"] += sumw
                    h["counts"] += counts

        # -----------------------------------------------------
        # Nominal: fill all cumulative cuts
        # -----------------------------------------------------
        else:

            cumulative_cuts = []

            for cut_name in self.cuts_to_store:

                cumulative_cuts.append(cut_name)

                mask = self.selections.all(*cumulative_cuts)
                weights = self.weights[mask]

                for category, variables in self.histogram_list.items():
                    for var_name, info in variables.items():

                        data = eval(
                            info["expression"],
                            {
                                "objects": self.objects,
                                "ak": ak,
                                "np": np,
                            },
                        )[mask]

                        h = self.histograms[var_name][cut_name]

                        sumw, counts = fill_with_overflow(
                            data,
                            weights,
                            h["edges"],
                        )

                        h["sumw"] += sumw
                        h["counts"] += counts

        return self.histograms