import numpy as np
import awkward as ak
from coffea.analysis_tools import Weights


def add_TopPtReweighting(
    events,
    weights_container: Weights,
    dataset: str,
    variation: str = "nominal",
):
    """
    Add top-pT reweight (nominal, up, down) to weights_container for ttbar MC.
    Uses GenPart.statusFlags bit 13 (isLastCopy) and pdgId==±6.
    Keeps event structure with ak.mask / ak.pad_none / ak.fill_none / ak.where.

    Parameters
    ----------
    events : NanoAOD events (awkward array)
    weights_container : coffea.analysis_tools.Weights

    dataset : str
        Dataset name (we apply only if dataset.startswith("TT"))

    params : dict
        Dictionary with keys: https://github.com/DesyTau/CPinHToTauTau/blob/ba5936185dbed76ccc58747ad84c4bf90b3168d6/httcp/production/top_pt_weight.py#L70
            {"a": 0.0615, "a_up": 0.0615*1.5, "a_down": 0.0615*0.5,
             "b": -0.0005, "b_up": -0.0005*1.5, "b_down": -0.0005*0.5}

            weight:
                w = exp^{a - b*pt}, with a = 0.0615, b = 0.0005 (data/POWHEG+Pythia8)
                Example: TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v1

    variation : str
        'nominal' by default, can be 'up' or 'down' (optional)

    - Documentation:

        GenPart_statusFlags, bits are: 
            0 : isPrompt, (N = 1)
            1 : isDecayedLeptonHadron,  (N = 2)
            2 : isTauDecayProduct,  (N = 4) 
            3 : isPromptTauDecayProduct, (N = 8)
            4 : isDirectTauDecayProduct, (N = 16)
            5 : isDirectPromptTauDecayProduct, (N = 32)
            6 : isDirectHadronDecayProduct, (N = 64)
            7 : isHardProcess, (N = 128)
            8 : fromHardProcess, (N = 256)
            9 : isHardProcessTauDecayProduct, (N = 512)
            10 : isDirectHardProcessTauDecayProduct, (N = 1024)
            11 : fromHardProcessBeforeFSR, (N = 2048)
            12 : isFirstCopy, (N = 4096)
            13 : isLastCopy, (N = 8192)
            14 : isLastCopyBeforeFSR, (N = 16384)

    bit 13 indicates whether the particle is the last copy in the event record (after radiation).

    References
    ----------
    https://twiki.cern.ch/twiki/bin/view/CMS/TopPtReweighting
    https://github.com/tanmayvb/ExoPieUtils/blob/746f7adce260730ca589fff8d75fe09db4154598/analysisutils/weight_utils.py#L114
    https://github.com/DesyTau/CPinHToTauTau/blob/ba5936185dbed76ccc58747ad84c4bf90b3168d6/httcp/production/top_pt_weight.py#L70
    https://github.com/cmantill/boostedhiggs/blob/3550092d3f740d2449660fe8331c12f796a15565/boostedhiggs/corrections.py#L908    
    """

    # default parameters if not provided
    params = {
        "a": 0.0615,
        "a_up": 0.0615 * 1.5,
        "a_down": 0.0615 * 0.5,
        "b": -0.0005,
        "b_up": -0.0005 * 1.5,
        "b_down": -0.0005 * 0.5,
    }

    # --- Apply only to ttbar ---
    if not dataset.startswith("TT"):
        ones = np.ones(len(events), dtype=float)
        weights_container.add("TopPtWeight", 
                              weight=ones,
                              weightUp=ones,
                              weightDown=ones)
        return

    # --- Select gen-level tops ---
    gen = events.GenPart
    is_lastcopy = (gen.statusFlags & (1 << 13)) != 0  # bit 13 = isLastCopy
    is_top = (abs(gen.pdgId) == 6)
    
    tops= gen[(is_top & is_lastcopy)]

    # --- Individual tops ----
    tops_padded = ak.pad_none(tops, 2)

    top = tops_padded[tops_padded.pdgId == 6]
    antitop = tops_padded[tops_padded.pdgId == -6]

    # --- SF calculation ---
    def sf(a, b, pt):
        return np.exp(a + b * pt)        
    
    # --- Dictionary to loop over variations ---
    variations = {
        "nominal": ("a", "b"),
        "up": ("a_up", "b_up"),
        "down": ("a_down", "b_down"),
    }

    weights = {}
    for var, (a_key, b_key) in variations.items():
        sf_top = sf(params[a_key], params[b_key], top.pt)
        sf_antitop = sf(params[a_key], params[b_key], antitop.pt)
        # Combine top and antitop
        weights[var] = ak.flatten(np.sqrt(sf_top * sf_antitop))

    # --- Add to weights container ---
    weights_container.add(
        "TopPtReweighting",
        weight=weights["nominal"],
        weightUp=weights["up"],
        weightDown=weights["down"],
    )