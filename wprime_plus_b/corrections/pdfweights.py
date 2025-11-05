import re
import numpy as np
import awkward as ak
from coffea.analysis_tools import Weights


def add_pdf_weight(
    events,
    weights_container: Weights,
    year: str = "2017",
    variation: str = "nominal",
    dataset: str = "",
):
    """
    Add PDF, αs, and combined (PDF⊕αs) systematic weight variations to the Coffea Weights container.

    Handles both standard (103 weights) and reduced (101 weights, no αs) PDF sets.

    This function follows the CMS MC Contact Report (Antonis Agapitos et al., B2G Meeting, 14 July 2020):

        δ_pdf   = sqrt( Σ_{k=1}^{100} (σ^(k) - σ^(0))² )
        δ_αs    = |σ^(102) - σ^(101)| / 2   [if αs weights exist]
        δ_total = sqrt( δ_pdf² + δ_αs² )

    where:
      • mem = 0       → central PDF (αs = 0.118)
      • mem = 1–100   → PDF eigenvector members
      • mem = 101–102 → αs variations (αs = 0.116, 0.120)

    The LHEPdfWeight branch contains ratios (w_var / w_nominal) for LHA IDs 306000–306102.
    Some samples (e.g. ST_s-channel_4f_leptonDecays) were produced with only 101 members
    and therefore lack the αs variations.

    The function computes and registers:
      • "PDFweight"       → PDF uncertainty only
      • "AlphaSweight"    → αs uncertainty only (if available)
      • "PDFAlphaSweight" → combined PDF⊕αs uncertainty (if available)

    References:
      - CMS MC Contact Report v3, slides 4–5 & 25–28:
        https://indico.cern.ch/event/938672/contributions/3943718/attachments/2073936/3482265/MC_ContactReport_v3.pdf
      - PDF4LHC recommendations, arXiv:1510.03865
    """


    # Remove trailing _X from dataset name
    cleaned_dataset = re.sub(r'_\d+$', '', dataset)

    # List of datasets that do not require PDF/αs variations
    skip_datasets = ["WW", "WZ", "ZZ"]

    # Case 1: Skip datasets → all weights = 1
    if any(cleaned_dataset.startswith(ds) for ds in skip_datasets):
        n_events = len(events.MET)
        w_central = np.ones_like(events.MET.pt)
        weights_container.add("PDF", weight=w_central, weightUp=w_central, weightDown=w_central)
        weights_container.add("Alpha(PDF)", weight=w_central, weightUp=w_central, weightDown=w_central)
        weights_container.add("PDFandAlpha", weight=w_central, weightUp=w_central, weightDown=w_central)
        return

    # Otherwise, perform PDF/αs calculations
    pdfweights = events.LHEPdfWeight
    nmem = ak.num(pdfweights)

    # Case 2: 103 members → PDF + αs
    if ak.all(nmem == 103):
        w0 = pdfweights[:, 0]        # central (mem=0)
        w_all = pdfweights[:, 1:101] # mem=1..100
        diffs = (w_all - w0[:, None]) ** 2
        delta_pdf = np.sqrt(ak.sum(diffs, axis=1))

        w_central = ak.ones_like(delta_pdf)
        w_up_pdf = w_central + delta_pdf
        w_down_pdf = w_central - delta_pdf

        weights_container.add(
            "PDF",
            weight=w_central,
            weightUp=w_up_pdf,
            weightDown=w_down_pdf,
        )

        # αs
        w_as_low = pdfweights[:, 101]
        w_as_high = pdfweights[:, 102]
        delta_alpha = 0.5 * np.abs(w_as_high - w_as_low)
        w_up_alpha = w_central + delta_alpha
        w_down_alpha = w_central - delta_alpha

        weights_container.add(
            "Alpha(PDF)",
            weight=w_central,
            weightUp=w_up_alpha,
            weightDown=w_down_alpha,
        )

        # Combined total
        delta_total = np.sqrt(delta_pdf**2 + delta_alpha**2)
        w_up_total = w_central + delta_total
        w_down_total = w_central - delta_total

        weights_container.add(
            "PDFandAlpha",
            weight=w_central,
            weightUp=w_up_total,
            weightDown=w_down_total,
        )

    # Case 3: 101 members → PDF only, αs = 1
    elif ak.all(nmem == 101):
        w0 = pdfweights[:, 0]
        w_all = pdfweights[:, 1:101]
        diffs = (w_all - w0[:, None]) ** 2
        delta_pdf = np.sqrt(ak.sum(diffs, axis=1))

        w_central = ak.ones_like(delta_pdf)
        w_up_pdf = w_central + delta_pdf
        w_down_pdf = w_central - delta_pdf

        weights_container.add(
            "PDF",
            weight=w_central,
            weightUp=w_up_pdf,
            weightDown=w_down_pdf,
        )

        # αs missing → fill with 1
        w_alpha = ak.ones_like(delta_pdf)
        weights_container.add(
            "Alpha(PDF)",
            weight=w_alpha,
            weightUp=w_alpha,
            weightDown=w_alpha,
        )

        # PDF + alpha = PDF only
        weights_container.add(
            "PDFandAlpha",
            weight=w_central,
            weightUp=w_up_pdf,
            weightDown=w_down_pdf,
        )

    else:
        raise RuntimeError(
            f"Unexpected number of PDF members: found {ak.to_list(np.unique(nmem))}, expected 101 or 103."
        )