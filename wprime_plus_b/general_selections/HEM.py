import numpy as np
import awkward as ak


def get_HEM_cleaning(events: ak.Array, 
            jets: ak.Array, 
            electrons: ak.Array, 
            year: str):
    """
    Apply the HEM (Hadronic Endcap Minus) veto for 2018 MC.

    The HEM issue affects part of the 2018 detector, producing fake MET
    when jets or electrons fall in a problematic detector region.

    For data:
        - The veto is applied only for runs >= 319077 (Run 2018C/D).

    For MC:
        - The veto is applied randomly with a probability corresponding
          to the integrated luminosity fraction of Run C+D (~63.2%).

    References:
        https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/2000.html

    Parameters
    ----------
    events : ak.Array
        NanoEvents array containing event-level information (must include `run`).

    jets : ak.Array
        Jet collection used to apply the HEM veto (AK4 jets).

    electrons : ak.Array
        Electron collection used to apply the HEM veto.

    year : str
        Data-taking year. The HEM veto is applied only for "2018".

    Returns
    -------
    ak.Array
        Boolean mask per event. `True` means the event PASSES the HEM veto.
    """

   # ------------------------------------------------------------
    # Default: no HEM veto for non-2018 data
    # ------------------------------------------------------------
    if year != "2018":
        return np.ones(len(events), dtype=bool)

    # ------------------------------------------------------------
    # Define the HEM problematic region
    # ------------------------------------------------------------
    hem_veto = ak.any(
        (
            (jets.eta > -3.2)
            & (jets.eta < -1.3)
            & (jets.phi > -1.57)
            & (jets.phi < -0.87)
        ),
        -1,
    ) | ak.any(
        (
            (electrons.pt > 30)
            & (electrons.eta > -3.2)
            & (electrons.eta < -1.3)
            & (electrons.phi > -1.57)
            & (electrons.phi < -0.87)
        ),
        -1,
    )
    hem_cleaning = (
        (
            (events.run >= 319077) & (not hasattr(events, "genWeight"))
        )  # if data check if in Runs C or D
        # else for MC randomly cut based on lumi fraction of C&D
        | ((np.random.rand(len(events)) < 0.632) & hasattr(events, "genWeight"))
    ) & (hem_veto)

    HEM_cleaning_mask = ~hem_cleaning
        

    return HEM_cleaning_mask