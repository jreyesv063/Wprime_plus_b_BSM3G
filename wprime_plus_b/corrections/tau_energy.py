import copy
import correctionlib
import numpy as np
import awkward as ak
from wprime_plus_b.corrections.utils import get_pog_json
from wprime_plus_b.corrections.met import update_met

# ----------------------------------------------------------------------------------- #
# -- The tau energy scale (TES) corrections for taus are provided  ------------------ #
# --  to be applied to reconstructed tau_h Lorentz vector ----------------------------#
# --  It should be applied to a genuine tau -> genmatch = 5 --------------------------#
# -----------  (pT, mass and energy) in simulated data -------------------------------#
# tau_E  *= tes
# tau_pt *= tes
# tau_m  *= tes
# https://github.com/cms-tau-pog/TauIDsfs/tree/master
# ----------------------------------------------------------------------------------- #


def mask_energy_corrections(tau):
    # https://github.com/cms-tau-pog/TauFW/blob/4056e9dec257b9f68d1a729c00aecc8e3e6bf97d/PicoProducer/python/analysis/ETauFakeRate/ModuleETau.py#L320
    # https://gitlab.cern.ch/cms-tau-pog/jsonpog-integration/-/blob/TauPOG_v2/POG/TAU/scripts/tau_tes.py

    tau_mask_gm = (
        (tau.genPartFlav == 5)  # Genuine tau
        | (tau.genPartFlav == 1)  # e -> fake
        | (tau.genPartFlav == 2)  # mu -> fake
        | (tau.genPartFlav == 6)  # unmached
    )
    tau_mask_dm = (
        (tau.decayMode == 0)
        | (tau.decayMode == 1)  # 1 prong
        | (tau.decayMode == 2)  # 1 prong
        | (tau.decayMode == 10)  # 1 prong
        | (tau.decayMode == 11)  # 3 prongs  # 3 prongs
    )
    tau_eta_mask = (tau.eta >= 0) & (tau.eta < 2.5)
    tau_mask = tau_mask_gm & tau_mask_dm  # & tau_eta_mask
    return tau_mask


def apply_tau_energy_scale_corrections(
    events: ak.Array,
    year: str = "2017",
    variation: str = "nominal",
):
    # define tau pt_raw field
    events["Tau", "pt_raw"] = events.Tau.pt

    # Flatten taus and apply mask
    ntaus = ak.num(events.Tau)
    taus_flatten = ak.flatten(events.Tau)
    mask = mask_energy_corrections(taus_flatten)
    taus_filter = taus_flatten.mask[mask]

    # Fill None values and get scale factors
    pt, eta, dm, genmatch = (ak.fill_none(taus_filter[field], 0) for field in ["pt", "eta", "decayMode", "genPartFlav"])
    cset = correctionlib.CorrectionSet.from_file(get_pog_json(json_name="tau", year=year))
    sf = {var: cset["tau_energy_scale"].evaluate(pt, eta, dm, genmatch, "DeepTau2017v2p1", var) for var in ["nom", "up", "down"]}


    # Compute new pt and mass values
    taus_new = {var: (taus_filter.pt * sf[var], taus_filter.mass * sf[var]) for var in sf}

    # Create corrected arrays with the same size as taus_flatten
    taus_corrected = {
        var: (
            ak.where(mask, taus_new[var][0], taus_flatten.pt),  # Corrected pt
            ak.where(mask, taus_new[var][1], taus_flatten.mass)  # Corrected mass
        )
        for var in sf
    }


    # Unflatten and update events
    for var, (pt, mass) in taus_corrected.items():
        events["Tau", f"pt{'_' + var if var != 'nom' else ''}"] = ak.unflatten(pt, ntaus)
        events["Tau", f"mass{'_' + var if var != 'nom' else ''}"] = ak.unflatten(mass, ntaus)

    # Propagate tau pT corrections to MET
    update_met(events=events, lepton="Tau")

   

    """
    # corrections works with flatten values
    ntaus = ak.num(copy.deepcopy(events.Tau))
    taus_flatten = ak.flatten(copy.deepcopy(events.Tau))

    # it is defined the taus will be corrected with the energy scale factor: Only a subset of the initial taus.
    mask = mask_energy_corrections(taus_flatten)
    taus_filter = taus_flatten.mask[mask]

    # fill None values with valid entries
    pt = ak.fill_none(taus_filter.pt, 0)
    eta = ak.fill_none(taus_filter.eta, 0)
    dm = ak.fill_none(taus_filter.decayMode, 0)
    genmatch = ak.fill_none(taus_filter.genPartFlav, 2)

    # define correction set
    cset = correctionlib.CorrectionSet.from_file(
        get_pog_json(json_name="tau", year=year)
    )

    # get scale factor
    sf = {
        "nominal": cset["tau_energy_scale"].evaluate(
            pt, eta, dm, genmatch, "DeepTau2017v2p1", "nom"
        ),
        "up": cset["tau_energy_scale"].evaluate(
            pt, eta, dm, genmatch, "DeepTau2017v2p1", "up"
        ),
        "down": cset["tau_energy_scale"].evaluate(
            pt, eta, dm, genmatch, "DeepTau2017v2p1", "down"
        ),
    }


    # get new (pT, mass) values using the scale factor
    taus_new_pt = taus_filter.pt * sf[variation]
    taus_new_mass = taus_filter.mass * sf[variation]
    new_tau_pt = ak.where(mask, taus_new_pt, taus_flatten.pt)
    new_tau_mass = ak.where(mask, taus_new_mass, taus_flatten.mass)

    # unflatten
    tau_pt = ak.unflatten(new_tau_pt, ntaus)
    tau_mass = ak.unflatten(new_tau_mass, ntaus)

    # update tau pt and mass fields
    events["Tau", "pt"] = tau_pt
    events["Tau", "mass"] = tau_mass

    # propagate tau pT corrections to MET
    update_met(events=events, lepton="Tau")
    """