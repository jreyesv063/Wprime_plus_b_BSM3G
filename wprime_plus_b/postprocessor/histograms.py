import hist

#---------------------------------
# ttbar control regions histograms
#---------------------------------
# jet hist
jet_pt_axis = hist.axis.Variable(
    edges=[30, 60, 90, 120, 150, 180, 210, 240, 300, 500],
    name="jet_pt",
)
jet_eta_axis = hist.axis.Regular(
    bins=50,
    start=-2.4,
    stop=2.4,
    name="jet_eta",
)
jet_phi_axis = hist.axis.Regular(
    bins=50,
    start=-4.0,
    stop=4.0,
    name="jet_phi",
)
jet_hist = hist.Hist(
    jet_pt_axis,
    jet_eta_axis,
    jet_phi_axis,
    hist.storage.Weight()
)
# met hist
met_axis = hist.axis.Variable(
    edges=[50, 75, 100, 125, 150, 175, 200, 300, 500],
    name="met",
)
met_phi_axis = hist.axis.Regular(
    bins=50,
    start=-4.0,
    stop=4.0,
    name="met_phi",
)
met_hist = hist.Hist(
    met_axis,
    met_phi_axis,
    hist.storage.Weight(),
)
# electron hist
lepton_pt_axis = hist.axis.Variable(
    edges=[30, 60, 90, 120, 150, 180, 210, 240, 300, 500],
    name="lepton_pt",
)
lepton_eta_axis = hist.axis.Regular(
    bins=50,
    start=-2.4,
    stop=2.4,
    name="lepton_eta",
)
lepton_phi_axis = hist.axis.Regular(
    bins=50,
    start=-4.0,
    stop=4.0,
    name="lepton_phi",
)
lepton_reliso = hist.axis.Regular(
    bins=25,
    start=0,
    stop=1,
    name="lepton_reliso",
)
lepton_hist = hist.Hist(
    lepton_pt_axis,
    lepton_eta_axis,
    lepton_phi_axis,
    hist.storage.Weight(),
)
# lepton + bjet
lepton_bjet_dr_axis = hist.axis.Regular(
    bins=30,
    start=0,
    stop=5,
    name="lepton_bjet_dr",
)
lepton_bjet_mass_axis = hist.axis.Variable(
    edges=[40, 75, 100, 125, 150, 175, 200, 300, 500],
    name="lepton_bjet_mass",
)
lepton_bjet_hist = hist.Hist(
    lepton_bjet_dr_axis,
    lepton_bjet_mass_axis,
    hist.storage.Weight(),
)
# lepton + missing energy
lepton_met_mass_axis = hist.axis.Variable(
    edges=[40, 75, 100, 125, 150, 175, 200, 300, 500, 800],
    name="lepton_met_mass",
)
lepton_met_delta_phi_axis = hist.axis.Regular(
    bins=30, start=0, stop=4, name="lepton_met_delta_phi"
)
lepton_met_hist = hist.Hist(
    lepton_met_mass_axis,
    lepton_met_delta_phi_axis,
    hist.storage.Weight(),
)
# lepton + missing energy + bjet
lepton_met_bjet_mass_axis = hist.axis.Variable(
    edges=[40, 75, 100, 125, 150, 175, 200, 300, 500, 800],
    name="lepton_met_bjet_mass",
)
lepton_met_bjet_hist = hist.Hist(
    lepton_met_bjet_mass_axis,
    hist.storage.Weight(),
)

ttbar_cr_histograms = {
    "jet_kin": jet_hist,
    "met_kin": met_hist,
    "lepton_kin": lepton_hist,
    "lepton_bjet_kin": lepton_bjet_hist,
    "lepton_met_kin": lepton_met_hist,
    "lepton_met_bjet_kin": lepton_met_bjet_hist,
}

#-------------------------------
# ztoll control region histogram
#-------------------------------
dilepton_mass_axis = hist.axis.Regular(
    bins=40, 
    start=50, 
    stop=200, 
    name="dilepton_mass"
)
dilepton_mass_hist = hist.Hist(
    dilepton_mass_axis, 
    hist.storage.Weight()
)
dilepton_mass_histogram = {
    "dilepton_kin": dilepton_mass_hist
}