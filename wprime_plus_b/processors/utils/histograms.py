import hist
import numpy as np

# --------------------------
# histogram axes definition
# --------------------------
# systematics axis
syst_axis = hist.axis.StrCategory([], name="variation", growth=True)

# ---------
# bjets
# --------
bjet_pt_axis = hist.axis.Regular(
    bins = 500,
    start = 0,
    stop = 1000,
    name="bjet_pt",
)
bjet_eta_axis = hist.axis.Regular(
    bins=50,
    start=-2.4,
    stop=2.4,
    name="bjet_eta",
)
bjet_phi_axis = hist.axis.Regular(
    bins=50,
    start=-np.pi,
    stop=np.pi,
    name="bjet_phi",
)


# ---------
# jets
# ---------
jet_pt_axis = hist.axis.Regular(
    bins = 500,
    start = 0,
    stop = 1000,
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

# ---------------
# Leading jet
# ---------------

# jet axes
leading_jet_pt_axis = hist.axis.Regular(
    bins = 500,
    start = 0,
    stop = 1000,
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

# -----
# MET
# -----
ttbar_met_axis = hist.axis.Regular(
    bins = 500,
    start = 0,
    stop = 1000,
    name="met",
)

met_phi_axis = hist.axis.Regular(
    bins=50,
    start=-np.pi,
    stop=np.pi,
    name="met_phi",
)

# ---------
# Lepton
# ---------
lepton_pt_axis = hist.axis.Regular(
    bins = 500,
    start = 0,
    stop = 1000,
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

# ---------
#  Tau
# ---------
tau_genPartFlav_axis = hist.axis.Regular(
    bins = 7,
    start = 0,
    stop = 7,
    name="genPartFlav",
)

tau_decayMode_axis = hist.axis.Regular(
    bins = 12,
    start = 0,
    stop = 12,
    name="decayMode",
)

# [lower, upper)
tau_isolation_electrons_axis = hist.axis.Variable(
    edges=[0, 1, 3, 7, 15, 31, 63, 127, 255, 300],
    name="isolation_electrons",
)

tau_isolation_jets_axis = hist.axis.Variable(
    edges=[0, 1, 3, 7, 15, 31, 63, 127, 255, 300],
    name="isolation_jets",
)

tau_isolation_muons_axis = hist.axis.Variable(
    edges=[0, 1, 3, 7, 15, 20],
    name="isolation_muons",
)


# ------------------
# lepton + bjet axes
# ------------------
lepton_bjet_dr_axis = hist.axis.Regular(
    bins=30,
    start=0,
    stop=5,
    name="lepton_bjet_dr",
)
lepton_bjet_mass_axis = hist.axis.Regular(
    bins = 500,
    start = 0,
    stop = 1000,
    name="lepton_bjet_mass",
)

# ---------------------------
# lepton + missing energy axes
# ---------------------------
lepton_met_mass_axis = hist.axis.Regular(
    bins = 500,
    start = 0,
    stop = 1000,
    name="lepton_met_mass",
)
lepton_met_delta_phi_axis = hist.axis.Regular(
    bins=30, 
    start=0, 
    stop=4, 
    name="lepton_met_delta_phi"
)

# ------------------------------------
# lepton + missing energy + bjet axes
# ------------------------------------
lepton_met_bjet_mass_axis = hist.axis.Regular(
    bins = 500,
    start = 0,
    stop = 1000,
    name="lepton_met_bjet_mass",
)


# -----------
# N objects
# -----------
# number of jets 
n_jets_axis = hist.axis.Regular(
    bins=15,
    start=0,
    stop=15,
    name="njets",
)

n_jets_all_axis = hist.axis.Regular(
    bins=15,
    start=0,
    stop=15,
    name="njets_full",
)
# number of bjets 
n_bjets_axis = hist.axis.Regular(
    bins=15,
    start=0,
    stop=15,
    name="nbjets",
)
# number of primary vertices
n_vertices_axis = hist.axis.Regular(
    bins=60,
    start=0,
    stop=60,
    name="npvs",
)
# number of muons
n_muons_axis = hist.axis.Regular(
    bins=15,
    start=0,
    stop=15,
    name="nmuons",
)
# number of electrons
n_electrons_axis = hist.axis.Regular(
    bins=15,
    start=0,
    stop=15,
    name="nelectrons",
)
# number of taus
n_taus_axis = hist.axis.Regular(
    bins=15,
    start=0,
    stop=15,
    name="ntaus",
)

# -----------
# top mass
# -----------
top_mrec = hist.axis.Regular(
    bins = 1000,
    start = 0,
    stop = 1000,
    name="top_mrec",
)


# -------
# HT and ST
# -------
ht_axis = hist.axis.Regular(
    bins = 1000,
    start = 0,
    stop = 2000,
    name="HT",
)


st_axis = hist.axis.Regular(
    bins = 1000,
    start = 0,
    stop = 2000,
    name="ST",
)

st_met_axis = hist.axis.Regular(
    bins = 1000,
    start = 0,
    stop = 2000,
    name="ST_met",
)

st_full_axis = hist.axis.Regular(
    bins = 1000,
    start = 0,
    stop = 2000,
    name="ST_full",
)



# -------------------------------------
# ttbar analysis histograms: Histograms
# -------------------------------------
# bjet histogram
ttbar_bjet_hist = hist.Hist(
    bjet_pt_axis,                # bjet_pt
    bjet_eta_axis,               # bjet_eta
    bjet_phi_axis,               # bjet_phi
    syst_axis,
    hist.storage.Weight(),
)

# jet histogram
ttbar_jet_hist = hist.Hist(
    jet_pt_axis,                 # jet_pt
    jet_eta_axis,                # eta_pt
    jet_phi_axis,                # phi_pt
    syst_axis,
    hist.storage.Weight(),
)

# leading jet histogram
leading_jet_hist = hist.Hist(
    leading_jet_pt_axis,   # "leading_jet_pt",
    leading_jet_eta_axis,  # "leading_jet_eta",
    leading_jet_phi_axis,  # "leading_jet_phi",
    syst_axis,
    hist.storage.Weight(),
)


# ST and HT histogram
st_ht_hist = hist.Hist(
    ht_axis,            # "HT"
    st_axis,            # "ST"
    st_met_axis,       # "ST_met"
    st_full_axis,       # "ST_full"
    syst_axis,                 # "variation"
    hist.storage.Weight(),
)


# met histogram
ttbar_met_hist = hist.Hist(
    ttbar_met_axis,             # met
    met_phi_axis,               # met_phi
    syst_axis,
    hist.storage.Weight(),
)

# lepton histogram
ttbar_lepton_hist = hist.Hist(
    lepton_pt_axis,            # lepton_pt
    lepton_eta_axis,           # lepton_eta
    lepton_phi_axis,           # lepton_phi
    syst_axis,
    hist.storage.Weight(),
)


# tau histogram
ttbar_tau_hist = hist.Hist(
    tau_genPartFlav_axis,         # genPartFlav
    tau_decayMode_axis,           # decayMode
    tau_isolation_electrons_axis, # isolation_electrons
    tau_isolation_jets_axis,      # isolation_jets
    tau_isolation_muons_axis,     # isolation_muons
    syst_axis,
    hist.storage.Weight(),
)

# lepton + bjet histogram
ttbar_lepton_bjet_hist = hist.Hist(
    lepton_bjet_dr_axis,        # lepton_bjet_dr
    lepton_bjet_mass_axis,      # lepton_bjet_mass
    syst_axis,
    hist.storage.Weight(),
)
# lepton + missing energy histogram
ttbar_lepton_met_hist = hist.Hist(
    lepton_met_mass_axis,       # lepton_met_mass
    lepton_met_delta_phi_axis,  # lepton_met_delta_phi
    syst_axis,
    hist.storage.Weight(),
)
# lepton + missing energy + bjet histogram
ttbar_lepton_met_bjet_hist = hist.Hist(
    lepton_met_bjet_mass_axis,  # lepton_met_bjet_mass
    syst_axis,
    hist.storage.Weight(),
)

# n objects
ttbar_n_hist = hist.Hist(
    n_jets_all_axis,           # njets_full
    n_jets_axis,               # njets
    n_bjets_axis,              # nbjets
    n_vertices_axis,           # npvs
    n_muons_axis,              # nmuons
    n_electrons_axis,          # nelectrons
    n_taus_axis,               # ntaus
    syst_axis,
    hist.storage.Weight(),
)

top_tagger_hist = hist.Hist(
    top_mrec,                  # "top_mrec"
    syst_axis,                 # "variation"
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

