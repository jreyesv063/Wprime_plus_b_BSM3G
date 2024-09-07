import hist
import numpy as np

# --------------------------
# histogram axes definition
# --------------------------
# systematics axis
syst_axis = hist.axis.StrCategory([], name="variation", growth=True)

# jet axes
jet_pt_axis = hist.axis.Variable(
    edges=[20, 60, 90, 120, 150, 180, 210, 240, 300, 500],
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
    start=-np.pi,
    stop=np.pi,
    name="jet_phi",
)
# met axes
ttbar_met_axis = hist.axis.Variable(
    edges=[50, 75, 100, 125, 150, 175, 200, 300, 500],
    name="met",
)

met_phi_axis = hist.axis.Regular(
    bins=50,
    start=-np.pi,
    stop=np.pi,
    name="met_phi",
)
# lepton axes
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
    start=-np.pi,
    stop=np.pi,
    name="lepton_phi",
)
lepton_reliso = hist.axis.Regular(
    bins=25,
    start=0,
    stop=1,
    name="lepton_reliso",
)
# lepton + bjet axes
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
# lepton + missing energy axes
lepton_met_mass_axis = hist.axis.Variable(
    edges=[40, 75, 100, 125, 150, 175, 200, 300, 500, 800],
    name="lepton_met_mass",
)
lepton_met_delta_phi_axis = hist.axis.Regular(
    bins=30, start=0, stop=4, name="lepton_met_delta_phi"
)
# lepton + missing energy + bjet axes
lepton_met_bjet_mass_axis = hist.axis.Variable(
    edges=[40, 75, 100, 125, 150, 175, 200, 300, 500, 800],
    name="lepton_met_bjet_mass",
)



# --------------------------
# ttbar analysis histograms
# --------------------------
# jet histogram
ttbar_jet_hist = hist.Hist(
    jet_pt_axis,
    jet_eta_axis,
    jet_phi_axis,
    syst_axis,
    hist.storage.Weight(),
)
# met histogram
ttbar_met_hist = hist.Hist(
    ttbar_met_axis,
    met_phi_axis,
    syst_axis,
    hist.storage.Weight(),
)
# lepton histogram
ttbar_lepton_hist = hist.Hist(
    lepton_pt_axis,
    lepton_eta_axis,
    lepton_phi_axis,
    syst_axis,
    hist.storage.Weight(),
)
# lepton + bjet histogram
ttbar_lepton_bjet_hist = hist.Hist(
    lepton_bjet_dr_axis,
    lepton_bjet_mass_axis,
    syst_axis,
    hist.storage.Weight(),
)
# lepton + missing energy histogram
ttbar_lepton_met_hist = hist.Hist(
    lepton_met_mass_axis,
    lepton_met_delta_phi_axis,
    syst_axis,
    hist.storage.Weight(),
)
# lepton + missing energy + bjet histogram
ttbar_lepton_met_bjet_hist = hist.Hist(
    lepton_met_bjet_mass_axis,
    syst_axis,
    hist.storage.Weight(),
)

# number of jets and primary vertices
n_jets_axis = hist.axis.Regular(
    bins=15,
    start=0,
    stop=15,
    name="njets",
)
n_vertices_axis = hist.axis.Regular(
    bins=60,
    start=0,
    stop=60,
    name="npvs",
)
ttbar_n_hist = hist.Hist(
    n_jets_axis,
    n_vertices_axis,
    syst_axis,
    hist.storage.Weight(),
)

# -------------------------------
# ztoll control region histogram
# -------------------------------
# lepton pt
ptl1_axis = hist.axis.Regular(
    bins=30, 
    start=20, 
    stop=200, 
    name="ptl1",
)
ptl1_histogram = hist.Hist(
    ptl1_axis,
    hist.storage.Weight(), 
)
ptl2_axis = hist.axis.Regular(
    bins=30, 
    start=20, 
    stop=200, 
    name="ptl2",
)
ptl2_histogram = hist.Hist(
    ptl2_axis,
    hist.storage.Weight(), 
)
# ptll 
ptll_axis = hist.axis.Regular(
    bins=30, 
    start=30, 
    stop=500, 
    name="ptll", 
    label="$p_T(ll)$ [GeV]"
)
ptll_histogram = hist.Hist(
    ptll_axis,
    hist.storage.Weight(), 
)

mll_axis = hist.axis.Regular(
    40, 30, 500.0, name="mll", label="$m_{ll}$ [GeV]"
)
mll_histogram = hist.Hist(
    mll_axis,
    hist.storage.Weight(), 
)

# ------------------------
# qcd analysis histograms
# ------------------------
# lepton ID region axis
lepton_id_axis = hist.axis.StrCategory([], name="lepton_id", growth=True)

qcd_met_axis = hist.axis.Variable(
    edges=[0, 50, 75, 100, 125, 150, 175, 200, 300, 500],
    name="met",
)

region_axis = hist.axis.StrCategory(["A", "B", "C", "D"], name="region")

# met histogram
qcd_met_hist = hist.Hist(
    qcd_met_axis,
    region_axis,
    hist.storage.Weight(),
)
# lepton + MET mass histogram
qcd_lepton_met_hist = hist.Hist(
    lepton_met_mass_axis,
    region_axis,
    hist.storage.Weight(),
)

# lepton + missing energy + bjet histogram
qcd_lepton_met_bjet_hist = hist.Hist(
    lepton_met_bjet_mass_axis,
    region_axis,
    hist.storage.Weight(),
)

# lepton + bjet histogram
qcd_lepton_bjet_hist = hist.Hist(
    lepton_bjet_mass_axis,
    region_axis,
    hist.storage.Weight(),
)

# -----------------------------
# Leading jet
# ------------------------------

# jet axes
leading_jet_pt_axis = hist.axis.Variable(
    edges=[20, 60, 90, 120, 150, 180, 210, 240, 300, 500],
    name="leading_jet_pt",
)
leading_jet_eta_axis = hist.axis.Regular(
    bins=50,
    start=-2.4,
    stop=2.4,
    name="leading_jet_eta",
)
leading_jet_phi_axis = hist.axis.Regular(
    bins=50,
    start=-np.pi,
    stop=np.pi,
    name="leading_jet_phi",
)

leading_jet_hist = hist.Hist(
    leading_jet_pt_axis,   # "leading_jet_pt",
    leading_jet_eta_axis,  # "leading_jet_eta",
    leading_jet_phi_axis,  # "leading_jet_phi",
    syst_axis,
    hist.storage.Weight(),
)


# jet axes
ht_axis = hist.axis.Variable(
    edges=[0, 70, 100, 200, 400, 600, 800, 1200, 2500, 50000],
    name="HT",
)
ht_hist = hist.Hist(
    ht_axis,   # "HT",
    syst_axis,
    hist.storage.Weight(),
)

# -----------------------------
# Z to ll
# -----------------------------
# m_rec axes
Z_mrec_pt_axis = hist.axis.Regular(
    bins = 1000,
    start = 0,
    stop = 1000,
    name="Z_mrec_pt",
)

# m_rec_MET axes
Z_mrec_mass_axis = hist.axis.Regular(
    bins = 1000,
    start = 0,
    stop = 1000,
    name="Z_mrec_mass",
)


# Z charge axes
Z_charge_axis = hist.axis.Variable(
    edges=[-2, -1, 0, 1, 2],
    name="Z_charge",
)

# lepton axes
leptonOne_pt_axis = hist.axis.Regular(
    bins = 1000,
    start = 0,
    stop = 1000,
    name="lepton_one_pt",
)
# lepton axes
leptonTwo_pt_axis = hist.axis.Regular(
    bins = 1000,
    start = 0,
    stop = 1000,
    name="lepton_two_pt",
)

Ztoll_hist = hist.Hist(
    Z_mrec_pt_axis,            # "Z_mrec_pt"
    Z_mrec_mass_axis,          # "Z_mrec_mass"
    Z_charge_axis,             # "Z_charge"
    leptonOne_pt_axis,         # "lepton_one_pt" 
    leptonTwo_pt_axis,         # "lepton_two_pt" 
    syst_axis,                 # "variation"
    hist.storage.Weight(),
)

# -----------------------------
# Top tagger
# ------------------------------
# top tagger histograms
# top axes
top_mrec = hist.axis.Regular(
    bins = 1000,
    start = 0,
    stop = 1000,
    name="top_mrec",
)
top_tagger_hist = hist.Hist(
    top_mrec,                  # "top_mrec"
    syst_axis,                 # "variation"
    hist.storage.Weight(),
)