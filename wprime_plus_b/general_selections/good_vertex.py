import awkward as ak

def get_good_vertex_mask(events: ak.Array) -> ak.Array:
    """
    Build a boolean mask for events with at least one good primary vertex.

    A good primary vertex is defined as an event having `npvsGood > 0`.

    Parameters
    ----------
    events : ak.Array
        NanoEvents array containing primary vertex information (events.PV).

    Returns
    -------
    ak.Array
        Boolean mask per event. True if the event has at least one good vertex.
    """
    if not hasattr(events.PV, "npvsGood"):
        raise AttributeError("The events.PV collection does not contain 'npvsGood'.")

    good_vertex_mask = events.PV.npvsGood > 0
    return good_vertex_mask
