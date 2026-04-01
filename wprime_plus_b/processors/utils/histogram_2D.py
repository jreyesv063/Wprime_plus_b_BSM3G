import numpy as np
import awkward as ak

def ISR_boost_plot(objects, out, mask, weights_container):
    
    Z_pt = objects['events'].Z_pt[mask]
    njets = ak.num(objects["jets"])[mask]

    weights = weights_container[mask]

    
    binning_Z = np.array([
        0, 20, 40, 60, 80, 100, 120, 150,
        200, 300, 400, 500, 600, 700,
        800, 900, 1000, 1500, 2000, 3000, 4000, 5000
    ])

    binning_nj = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


    
    H, Z_pt_edges, nj_edges = np.histogram2d(
        Z_pt,
        njets,
        bins=[binning_Z, binning_nj],
        weights=weights
    )


    out["hist"] = H
    out["X_edges"] = Z_pt_edges
    out["Y_edges"] = nj_edges


def ttbar_boost_plot(objects, out, mask, weights_container, lepton_flavor):


    lepton_map = {"ele": objects["electrons"], "mu": objects["muons"], "tau": objects["taus"]}


    lepton = ak.firsts(lepton_map[lepton_flavor][mask])
    met = objects['met'][mask]
    events = objects["events"][mask]
    weights = ak.to_numpy(weights_container[mask])
    
  
    ST = events.top_tagger_pt + lepton.pt + met.pt
    
    nj = ak.to_numpy(events.njets_noTopTagger)

    
    binning_ST = np.array([
        0, 100, 150,
        200, 300, 400, 500, 600, 700,
        800, 900, 1000, 1500, 2000, 3000, 4000, 5000
    ])

    binning_nj = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


    
    H, ST_edges, nj_edges = np.histogram2d(
        ST,
        nj,
        bins=[binning_ST, binning_nj],
        weights=weights
    )



    out["hist"] =  H
    out["X_edges"] = ST_edges
    out["Y_edges"] = nj_edges

