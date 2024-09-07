import numpy as np
import awkward as ak
from wprime_plus_b.corrections.met import update_met
from coffea.lookup_tools import txt_converters, rochester_lookup


def apply_rochester_corrections(
    events: ak.Array, is_mc: bool, year: str = "2017", variation: str = "nominal"
):
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

        if variation != "nominal":
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
        if variation != "nominal":
            errors = rochester.kScaleDTerror(
                events.Muon.charge, events.Muon.pt, events.Muon.eta, events.Muon.phi
            )
        
    pt_shift = {
        "nominal": events.Muon.pt * corrections,
    }
    if variation != "nominal":
        pt_shift.update(
            {
                "rochester_up": events.Muon.pt * corrections + events.Muon.pt * errors,
                "rochester_down": events.Muon.pt * corrections
                - events.Muon.pt * errors,
            }
        )
    # get rochester pT for Muon
    events["Muon", "pt_rochester"] = pt_shift[variation]
    
    # update muon pT field
    events["Muon", "pt"] = events.Muon.pt_rochester
    
    # propagate muon pT corrections to MET
    update_met(events=events, lepton="Muon")