import numpy as np
import awkward as ak
from wprime_plus_b.corrections.met import update_met, update_met_list
from coffea.lookup_tools import txt_converters, rochester_lookup


def apply_rochester_corrections(
    events: ak.Array, is_mc: bool, year: str = "2017", variation: bool = False):
    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/RochcorMuon
    rochester_data = txt_converters.convert_rochester_file(
        f"wprime_plus_b/data/RoccoR{year}UL.txt", loaduncs=True
    )
    rochester = rochester_lookup.rochester_lookup(rochester_data)

    # define muon pt_raw field
    events["Muon", "pt_raw"] = ak.ones_like(events.Muon.pt) * events.Muon.pt

    if is_mc:
        hasgen = ~np.isnan(ak.fill_none(events.Muon.matched_gen.pt, np.nan))
        mc_rand = np.random.rand(*ak.to_numpy(ak.flatten(events.Muon.pt)).shape)
        mc_rand = ak.unflatten(mc_rand, ak.num(events.Muon.pt, axis=1))
        corrections = np.array(ak.flatten(ak.ones_like(events.Muon.pt)))
        mc_kspread = rochester.kSpreadMC(
            events.Muon.charge[hasgen],
            events.Muon.pt[hasgen],
            events.Muon.eta[hasgen],
            events.Muon.phi[hasgen],
            events.Muon.matched_gen.pt[hasgen],
        )
        mc_ksmear = rochester.kSmearMC(
            events.Muon.charge[~hasgen],
            events.Muon.pt[~hasgen],
            events.Muon.eta[~hasgen],
            events.Muon.phi[~hasgen],
            events.Muon.nTrackerLayers[~hasgen],
            mc_rand[~hasgen],
        )
        hasgen_flat = np.array(ak.flatten(hasgen))
        corrections[hasgen_flat] = np.array(ak.flatten(mc_kspread))
        corrections[~hasgen_flat] = np.array(ak.flatten(mc_ksmear))
        corrections = ak.unflatten(corrections, ak.num(events.Muon.pt, axis=1))

        errors = np.array(ak.flatten(ak.ones_like(events.Muon.pt)))
        errspread = rochester.kSpreadMCerror(
            events.Muon.charge[hasgen],
            events.Muon.pt[hasgen],
            events.Muon.eta[hasgen],
            events.Muon.phi[hasgen],
            events.Muon.matched_gen.pt[hasgen],
        )
        errsmear = rochester.kSmearMCerror(
            events.Muon.charge[~hasgen],
            events.Muon.pt[~hasgen],
            events.Muon.eta[~hasgen],
            events.Muon.phi[~hasgen],
            events.Muon.nTrackerLayers[~hasgen],
            mc_rand[~hasgen],
        )
        errors[hasgen_flat] = np.array(ak.flatten(errspread))
        errors[~hasgen_flat] = np.array(ak.flatten(errsmear))
        errors = ak.unflatten(errors, ak.num(events.Muon.pt, axis=1))
    else:
        corrections = rochester.kScaleDT(
            events.Muon.charge, events.Muon.pt, events.Muon.eta, events.Muon.phi
        )

        errors = rochester.kScaleDTerror(
            events.Muon.charge, events.Muon.pt, events.Muon.eta, events.Muon.phi
        )

    # Compute and save pt shifts in events
    for var, shift in {
        "pt_rochester": corrections,
        "pt_up": corrections + errors,
        "pt_down": corrections - errors,
    }.items():
        events["Muon", var] = events.Muon.pt * shift

    # Update muon pt field
    events["Muon", "pt"] = events.Muon.pt_rochester

    if variation:
        
        muon_list = {
            "nom": events.Muon.pt,
            "up": events.Muon.pt_up,
            "down": events.Muon.pt_down,
            "phi": events.Muon.phi,
        }

        met_pt_list, met_phi_list, delta_list = update_met_list(   
            events=events,
            syst_name="Muon",
            syst_var = variation, 
            object = muon_list,       
        )

        return delta_list #met_pt_list, met_phi_list, delta_list
        
    else:
        update_met(events=events, lepton="Muon")
