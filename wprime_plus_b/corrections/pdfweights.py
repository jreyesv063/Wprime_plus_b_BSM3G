import re
import numpy as np
import awkward as ak
from coffea.analysis_tools import Weights


def add_pdf_weight(
    events: ak.Array,
    weights_container: Weights,
    output: dict = None,
    year: str = "2017",
):
    """
    Source 1: https://cms-opendata-guide.web.cern.ch/analysis/systematics/mcuncertain/#variations-of-generator-parameters
    Source 2: CMS MC Contact Report v3, slides 4–5 & 25–28: https://indico.cern.ch/event/938672/contributions/3943718/attachments/2073936/3482265/MC_ContactReport_v3.pdf
    Source 3: PDF4LHC recommendations, arXiv:1510.03865
    Source 4:  https://github.com/GhentAnalysis/columnflow_old/blob/924cb6f348811da3fd2647da15ce1c0b831c770a/columnflow/production/cms/pdf.py#L81

    Parton distribution functions (PDFs) describe the probability for finding, within a proton, a parton of a certain flavor and momentum fraction. The uncertainties in a PDF are provided by the collaborations that produce them according to two methods:

    - Monte Carlo replicas: Replicas are made of the PDF based on random variations of parameters.
    - Hessian uncertainties: Uncertainties in the PDF are factorized from each other. The deviation from the central PDF estimate is added in quadrature to the other variations (Our analysis).

    Following the documentation:
    
    - Run 2: LHEPdfWeight: 	LHE pdf variation weights (w_var / w_nominal) for LHA IDs 306000 - 306102 
    - Run 3: LHEPdfWeight:: LHE pdf variation weights (w_var / w_nominal) for LHA IDs 325300 - 325402



    Methodology:
    
        δ_pdf   = sqrt( Σ_{k=1}^{100} (σ^(k) - σ^(0))² )
        δ_αs    = |σ^(102) - σ^(101)| / 2   [if αs weights exist]
        δ_total = sqrt( δ_pdf² + δ_αs² )

    Important: the variations are relative to the nominal value. Therefore, if the delta is 3, it means that it is 3% of the nominal value.

    where:
      • mem = 0       → central PDF (αs = 0.118)
      • mem = 1–100   → PDF eigenvector members
      • mem = 101–102 → αs variations (αs = 0.116, 0.120)


    The function computes and registers:
      • "PDFweight"       → PDF uncertainty only
      • "AlphaSweight"    → αs uncertainty only (if available)
      • "PDFAlphaSweight" → combined PDF and αs uncertainty (if available)



    Important: 
    
    - Run 2: LHEPdfWeight: 	LHE pdf variation weights (w_var / w_nominal) for LHA IDs 306000 - 306102 
    - Run 3: LHEPdfWeight:: LHE pdf variation weights (w_var / w_nominal) for LHA IDs 325300 - 325402

    PDF and alpha_s weights: https://lhapdfsets.web.cern.ch/current/NNPDF31_nnlo_hessian_pdfas/NNPDF31_nnlo_hessian_pdfas.info -> ErrorType: symmhessian+as
    #(68% CL via percentiles, event-by-event stable)
    
    """
    # -----------------------------------------------------------------------------
    # PDF and Alpha_s Uncertainties for Symmetric Hessian Set
    # Reference: LHA ID 306000 (NNPDF31_nnlo_as_0118 variant)
    # ErrorType: symmhessian+as
    # -----------------------------------------------------------------------------
    
    # 1. Initialize default weights (nominal = 1.0)
    # Using float32 for consistency with NanoAOD precision
    ones = ak.ones_like(events.MET.pt)
    w_pdf_up = w_pdf_down = w_alpha_up = w_alpha_down = w_total_up = w_total_down = ones
    
    delta_pdf = pdf_weight_nominal = ones  # Default values in case weights are missing

    if hasattr(events, "LHEPdfWeight"):
        
        # IMPORTANT: Use values_astype to avoid "NanoCollection" type errors 
        # when performing arithmetic operations.
        pdf_weights_all = ak.values_astype(events.LHEPdfWeight, "float32")
        
        # Get number of weights per event
        n_weights = ak.num(pdf_weights_all, axis=1)
    
        # Consistency check
        if not (ak.all(n_weights == 101) or ak.all(n_weights == 103)):
            raise RuntimeError("Inconsistent LHEPdfWeight size. Expected 101 or 103.")
            
        # 2. Strict Validation: Ensure the nominal weight (index 0) is EXACTLY 1.0
        pdf_weight_nominal = pdf_weights_all[:, 0]
        
    
        # 3. PDF variations (Symmetric Hessian: indices 1 to 100)
        pdf_weights_vars = pdf_weights_all[:, 1:101]
        
        # Formula: delta_pdf = sqrt( sum_{i=1}^{100} (w_i - 1.0)^2 )
        # Subtracting 'ones' ensures proper broadcasting across the 100 variations
        deviations = pdf_weights_vars - pdf_weight_nominal
        delta_pdf = np.sqrt(ak.sum(deviations**2, axis=1))
        
        # Define PDF Up/Down variations with clipping at 0.0
        w_pdf_up = ak.where(delta_pdf > pdf_weight_nominal, pdf_weight_nominal, pdf_weight_nominal + delta_pdf)
        w_pdf_down = ak.where(delta_pdf > pdf_weight_nominal, pdf_weight_nominal, pdf_weight_nominal - delta_pdf)
    
        # 4. Alpha_s uncertainty (indices 101 and 102)
        if ak.all(n_weights == 103):
            alpha_one = pdf_weights_all[:, 101]
            alpha_two = pdf_weights_all[:, 102]
    
            delta_alpha = 0.5 * np.abs(alpha_one - alpha_two)
    
            w_alpha_up = ak.where(delta_alpha > pdf_weight_nominal, pdf_weight_nominal, pdf_weight_nominal + delta_alpha)
            w_alpha_down = ak.where(delta_alpha > pdf_weight_nominal, pdf_weight_nominal, pdf_weight_nominal - delta_alpha)
    
            # 5. Combined PDF + Alpha_s uncertainty in quadrature
            delta_total = np.sqrt(delta_pdf**2 + delta_alpha**2)
            w_total_up = ak.where(delta_total > pdf_weight_nominal, pdf_weight_nominal, pdf_weight_nominal + delta_total)
            w_total_down = ak.where(delta_total > pdf_weight_nominal, pdf_weight_nominal, pdf_weight_nominal - delta_total)
        else:
            delta_total = delta_pdf
            w_total_up = w_pdf_up
            w_total_down = w_pdf_down
    # -----------------------------------------------------------------------------
    # Register weights in the container
    # -----------------------------------------------------------------------------
    
    # 1. Pure PDF uncertainty (LHA variations only)
    weights_container.add(
        f"pdf_lha_{year}",
        weight=ones,
        weightUp=w_pdf_up,
        weightDown=w_pdf_down,
    )
    
    # 2. Pure Alpha_s uncertainty
    # (Only added if alpha_s variations were present in the input)
    weights_container.add(
        f"pdf_alphas_{year}",
        weight=ones,
        weightUp=w_alpha_up,
        weightDown=w_alpha_down,
    )
    
    # 3. Combined PDF + Alpha_s uncertainty
    # This is the one usually used as the final systematic "PDF" in results
    weights_container.add(
        f"pdf_lha_alphas_{year}",
        weight=ones,
        weightUp=w_total_up,
        weightDown=w_total_down,
    )
    return delta_pdf, pdf_weight_nominal