import yaml
import awkward as ak
import numpy as np


class Histograms:
    def __init__(self, lepton, processor, objects, weights, selections, cutflow_names, is_syst_var: bool = False,):
        self.objects = objects
        self.weights = weights
        self.selections = selections
        self.processor = processor
        self.lepton = lepton

        self.is_syst_var = is_syst_var

        if self.is_syst_var:
            self.cuts_to_store = [cutflow_names[-1]]  # Lista de nombres de cortes
        else:
            self.cuts_to_store = cutflow_names


        self.cutflow_names = cutflow_names
        
        # Aquí guardaremos SOLO arrays NumPy
        self.histograms = {}

        self.get_histogram_config()
        self.create_histograms()

    # ---------------------------------------------------------
    # Leer YAML y filtrar variables a plotear
    # ---------------------------------------------------------
    def get_histogram_config(self):
        with open(
            f"wprime_plus_b/selection_criteria/{self.processor}/Histograms_{self.lepton}.yaml", "r"
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

        # Eliminar categorías vacías
        self.histogram_list = {
            cat: vars for cat, vars in self.histogram_list.items() if vars
        }

    # ---------------------------------------------------------
    # Inicializar estructuras ligeras de histogramas
    # ---------------------------------------------------------
    def create_histograms(self):
        for category, variables in self.histogram_list.items():
            for var_name, info in variables.items():
                
                # Definimos edges
                if info["type"] == "Regular":
                    edges = np.linspace(
                        info["start"],
                        info["stop"],
                        info["bins"] + 1,
                        dtype=np.float64,
                    )
                else:
                    edges = np.asarray(info["edges"], dtype=np.float64)

                # Un histograma por variable y por corte
                self.histograms[var_name] = {
                    cut: {
                        "edges":  edges.copy(), #edges,
                        "sumw": np.zeros(len(edges) - 1, dtype=np.float64),
                        "sumw2": np.zeros(len(edges) - 1, dtype=np.float64),
                    }
                    for cut in self.cuts_to_store
                }

    # ---------------------------------------------------------
    # Llenado de histogramas
    # ---------------------------------------------------------
    def fill_histograms(self):
    
        def fill_with_overflow(data, weights, edges):
            # Histograma normal
            sumw, _ = np.histogram(data, bins=edges, weights=weights)
            sumw2, _ = np.histogram(data, bins=edges, weights=weights ** 2)
    
            # Underflow -> primer bin
            under_mask = data < edges[0]
            if np.any(under_mask):
                w = weights[under_mask]
                sumw[0] += np.sum(w)
                sumw2[0] += np.sum(w ** 2)
    
            # Overflow -> último bin
            over_mask = data >= edges[-1]
            if np.any(over_mask):
                w = weights[over_mask]
                sumw[-1] += np.sum(w)
                sumw2[-1] += np.sum(w ** 2)
    
            return sumw, sumw2
    
        # ------------------------------
        # Sistemáticas: solo último corte
        # ------------------------------
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
    
                    sumw, sumw2 = fill_with_overflow(
                        data,
                        weights,
                        h["edges"],
                    )
    
                    h["sumw"] += sumw
                    h["sumw2"] += sumw2
    
        # ------------------------------
        # Nominal: todos los cortes
        # ------------------------------
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
                            {"objects": self.objects, "ak": ak, "np": np},
                        )[mask]
    
                        h = self.histograms[var_name][cut_name]

                        sumw, sumw2 = fill_with_overflow(
                            data,
                            weights,
                            h["edges"],
                        )
    
                        h["sumw"] += sumw
                        h["sumw2"] += sumw2
    
        return self.histograms
