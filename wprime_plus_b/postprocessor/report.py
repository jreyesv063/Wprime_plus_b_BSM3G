import numpy as np
import pandas as pd


def build_report(
    accumulated_outputs: dict, xsecs: dict, lumi: float = 41477.877399
) -> pd.DataFrame:
    """
    Build a report containing the expected number of events and statistical errors for backgrounds,
    data, total background, and data/total background ratio

    Arguments:
        accumulated_outputs (dict): A dictionary containing the accumulated outputs for different samples.
        xsecs (dict): A dictionary containing the cross-sections for different samples.
        lumi (float): The luminosity value used for scaling the expected number of events (default: 41477.877399 (2017)).

    Returns:
        pd.DataFrame: A pandas DataFrame containing the report with columns 'events', 'error', and 'percentage'.
    """
    mcs = ["DYJetsToLL", "WJetsToLNu", "VV", "tt", "SingleTop", "Higgs"]
    events = {sample: 0 for sample in mcs}
    events.update({"Data": 0})
    errors = events.copy()

    for sample in accumulated_outputs:
        if ("SingleElectron" in sample) or ("SingleMuon" in sample):
            events["Data"] += accumulated_outputs[sample]["events_after"]
            errors["Data"] += np.sqrt(events["Data"])
            continue
        # get number of events before selection
        nevents = accumulated_outputs[sample]["events_before"]

        # get number of events after selection
        n_mc = accumulated_outputs[sample]["events_after"]

        # get expected number of events
        weight = (xsecs[sample] * lumi) / nevents
        n_phys = weight * n_mc

        # get statistical error
        error = weight * np.sqrt(n_mc)

        if "DYJetsToLL" in sample:
            events["DYJetsToLL"] += n_phys
            errors["DYJetsToLL"] += error
        elif "WJetsToLNu" in sample:
            events["WJetsToLNu"] += n_phys
            errors["WJetsToLNu"] += error
        elif (sample == "WW") or (sample == "WZ") or (sample == "ZZ"):
            events["VV"] += n_phys
            errors["VV"] += error
        elif "TTT" in sample:
            events["tt"] += n_phys
            errors["tt"] += error
        elif "ST" in sample:
            events["SingleTop"] += n_phys
            errors["SingleTop"] += error
        elif ("VBFH" in sample) or ("GluGluH" in sample):
            events["Higgs"] += n_phys
            errors["Higgs"] += error

    # add number of expected events and errors to report
    report_df = pd.DataFrame(columns=["events", "error", "percentage"])
    for sample in events:
        report_df.loc[sample, "events"] = events[sample]
        report_df.loc[sample, "error"] = errors[sample]

    # add percentages to report
    mcs_output = report_df.loc[mcs].copy()
    report_df.loc[mcs, "percentage"] = (
        mcs_output["events"] / mcs_output["events"].sum()
    ) * 100

    # (https://depts.washington.edu/imreslab/2011%20Lectures/ErrorProp-CountingStat_LRM_04Oct2011.pdf)
    # add total background number of expected events and error to report
    report_df.loc["Total bkg", "events"] = np.sum(report_df.loc[mcs, "events"])
    report_df.loc["Total bkg", "error"] = np.sqrt(
        np.sum(report_df.loc[mcs, "error"] ** 2)
    )

    # add data to bacground ratio and error
    data = report_df.loc["Data", "events"]
    data_err = report_df.loc["Data", "error"]
    bkg = report_df.loc["Total bkg", "events"]
    bkg_err = report_df.loc["Total bkg", "error"]

    report_df.loc["Data/bkg", "events"] = data / bkg
    report_df.loc["Data/bkg", "error"] = np.sqrt(
        (1 / bkg) ** 2 * data_err**2 + (data / bkg**2) ** 2 * bkg_err**2
    )
    # sort processes by percentage
    report_df = report_df.loc[mcs + ["Total bkg", "Data", "Data/bkg"]]
    report_df = report_df.sort_values(by="percentage", ascending=False)
    
    # drop process with no events
    report_df = report_df.loc[report_df.sum(axis=1) > 0]
    
    return report_df
