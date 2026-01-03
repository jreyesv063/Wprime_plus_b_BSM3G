import numpy as np
import awkward as ak


def select_good_mt(
    met: ak.Array,
    lepton: ak.Array,
    mt_cut: float,
    invert_mt_cut: bool = False,
) -> ak.Array:
    """
    Build an event-level boolean mask based on the transverse mass (mT)
    between a lepton and MET.

    Parameters
    ----------
    met : ak.Array
        MET collection (must provide pt and phi).

    lepton : ak.Array
        Lepton collection (must provide pt and delta_phi method).

    mt_cut : float
        Minimum transverse mass threshold.

    invert_mt_cut : bool, optional
        If True, invert the mT selection (i.e. select events failing the cut).

    Returns
    -------
    ak.Array
        Boolean mask at the event level.
    """

    # ============================================================
    # Compute transverse mass mT(lepton, MET)
    # mT = sqrt( 2 * pT_lep * pT_met * (1 - cos(Δφ)) )
    # ============================================================

    delta_phi = lepton.delta_phi(met)

    mt = np.sqrt(
        2.0
        * lepton.pt
        * met.pt
        * (1.0 - np.cos(delta_phi))
    )

    # ============================================================
    # Apply mT requirement at the lepton level
    # ============================================================

    lepton_pass_mt = mt >= mt_cut

    # ============================================================
    # Reduce to event level:
    # event passes if at least one lepton satisfies the mT cut
    # ============================================================

    event_pass_mt = ak.any(lepton_pass_mt, axis=-1)

    # ============================================================
    # Optional inversion of the selection
    # ============================================================
    if invert_mt_cut:
        return ~event_pass_mt

    return event_pass_mt
