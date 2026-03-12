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
    out["Z_pt_edges"] = Z_pt_edges
    out["nj_edges"] = nj_edges




def ttbar_boost_plot(objects, out, mask, weights_container, lepton_flavor):


    lepton_map = {"ele": objects["electrons"], "mu": objects["muons"], "tau": objects["taus"]}


    lepton = lepton_map[lepton_flavor][mask]
    met = objects['met'][mask]
    bjets = objects['bjets'][mask]
    lightjets = objects['lightjets'][mask]
    topjets = objects['topjets'][mask]
    wjets = objects['wjets'][mask]
    events = objects["events"][mask]
    weights = ak.to_numpy(weights_container[mask])
    
    ST = ak.to_numpy(
            ak.sum(lepton.pt, axis=1) 
            + met.pt 
            + ak.sum(bjets.pt, axis=1) 
            + ak.sum(lightjets.pt, axis=1) 
            + ak.sum(topjets.pt, axis=1) 
            + ak.sum(wjets.pt, axis=1)
    )
    
    nj = ak.to_numpy(events.njets_noTopTagger)

    
    binning_ST = np.array([
        0, 20, 40, 60, 80, 100, 120, 150,
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
    out["ST_edges"] = ST_edges
    out["nj_edges"] = nj_edges
