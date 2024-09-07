import glob
import copy
import pickle
import numpy as np
from coffea import processor


def open_output(output_fname: str) -> dict:
    """open .pkl output file"""
    with open(output_fname, "rb") as f:
        output = pickle.load(f)
    return output


def group_outputs(output_directory: str) -> dict:
    """group output .pkl files by sample"""
    output_files = glob.glob(f"{output_directory}/*.pkl", recursive=True)
    grouped_outputs = {}
    for output_file in output_files:
        # get output file names
        sample_name = output_file.split("/")[-1].split(".pkl")[0]
        if sample_name.rsplit("_")[-1].isdigit():
            sample_name = "_".join(sample_name.rsplit("_")[:-1])
        # append file names to grouped_outputs
        if sample_name in grouped_outputs:
            grouped_outputs[sample_name].append(output_file)
        else:
            grouped_outputs[sample_name] = [output_file]
    return grouped_outputs


def accumulate_outputs(grouped_outputs: dict) -> dict:
    """accumulate output arrays by sample"""
    accumulated_outputs = {}
    for sample in grouped_outputs:
        accumulated_outputs[sample] = []
        for output_fname in grouped_outputs[sample]:
            output = open_output(output_fname)
            accumulated_outputs[sample].append(output)
        accumulated_outputs[sample] = processor.accumulate(accumulated_outputs[sample])
    return accumulated_outputs


def fill_histograms(
    accumulated_outputs: dict, hist_histograms: dict, weighted=True
) -> dict:
    """fill hist histograms using accumulated outputs"""
    filled_histograms = {}
    for sample, values in accumulated_outputs.items():
        histograms = copy.deepcopy(hist_histograms)
        sample_weight = values["weights"].value
        weight = sample_weight if weighted else np.ones_like(sample_weight)
        filled_histograms[sample] = {}
        for kin in histograms:
            fill_args = {var: values[var].value for var in histograms[kin].axes.name}
            filled_histograms[sample][kin] = histograms[kin].fill(
                **fill_args, weight=weight
            )
    return filled_histograms


def get_lumiweights(
    accumulated_outputs: dict, xsecs: dict, lumi: float = 41477.877399, weighted=True
) -> dict:
    """compute luminosity-xsec weights"""
    sumws = {}
    for sample, values in accumulated_outputs.items():
        if sample in ["SingleMuon", "SingleElectron"]:
            continue
        sumws[sample] = values["sumw"] if weighted else values["events_before"]
    return {sample: lumi * xsecs[sample] / sumws[sample] for sample in sumws}


def scale_histograms(histograms: dict, lumi_weights: dict) -> dict:
    """scale histograms to luminosity-xsec weight"""
    scaled_histograms = {}
    for sample in histograms:
        scaled_histograms[sample] = {}
        for kin in histograms[sample]:
            histogram = copy.deepcopy(histograms[sample][kin])
            if sample in ["SingleMuon", "SingleElectron"]:
                scaled_histograms[sample][kin] = histogram
            else:
                scaled_histograms[sample][kin] = histogram * lumi_weights[sample]
    return scaled_histograms


def group_histograms(scaled_histograms: dict) -> dict:
    """group scaled histograms by process"""
    hists = {
        "DYJetsToLL": [],
        "WJetsToLNu": [],
        "VV": [],
        "tt": [],
        "SingleTop": [],
        "Higgs": [],
        "Data": [],
    }
    for sample in scaled_histograms:
        if "DYJetsToLL" in sample:
            hists["DYJetsToLL"].append(scaled_histograms[sample])
        elif "WJetsToLNu" in sample:
            hists["WJetsToLNu"].append(scaled_histograms[sample])
        elif (sample == "WW") or (sample == "WZ") or (sample == "ZZ"):
            hists["VV"].append(scaled_histograms[sample])
        elif "TTT" in sample:
            hists["tt"].append(scaled_histograms[sample])
        elif "ST" in sample:
            hists["SingleTop"].append(scaled_histograms[sample])
        elif ("VBFH" in sample) or ("GluGluH" in sample):
            hists["Higgs"].append(scaled_histograms[sample])
        else:
            hists["Data"] = scaled_histograms[sample]
    for sample in hists:
        if sample == "Data":
            continue
        hists[sample] = processor.accumulate(hists[sample])
    return hists


def get_mc_error(
    accumulated_outputs: dict,
    hist_histograms: dict,
    xsecs: dict,
    lumi: float = 41477.877399,
) -> dict:
    """compute statistical error for mc backgrounds"""
    histograms = fill_histograms(accumulated_outputs, hist_histograms, weighted=False)
    lumi_weights = get_lumiweights(
        accumulated_outputs, xsecs=xsecs, lumi=lumi, weighted=False
    )
    # compute mc error per variable as W_lumi * sqrt(N_mc)
    mc_errors = {}
    for sample in histograms:
        if sample in ["SingleMuon", "SingleElectron"]:
            continue
        mc_errors[sample] = {}
        for kin in histograms[sample]:
            mc_errors[sample][kin] = {}
            for var in histograms[sample][kin].axes.name:
                mc_errors[sample][kin][var] = lumi_weights[sample] * np.sqrt(
                    histograms[sample][kin].project(var).values()
                )
    
    hists = {
        "DYJetsToLL": [],
        "WJetsToLNu": [],
        "VV": [],
        "tt": [],
        "SingleTop": [],
        "Higgs": [],
    }
    # group mc errors by process
    for sample in mc_errors:
        if "DYJetsToLL" in sample:
            hists["DYJetsToLL"].append(mc_errors[sample])
        elif "WJetsToLNu" in sample:
            hists["WJetsToLNu"].append(mc_errors[sample])
        elif (sample == "WW") or (sample == "WZ") or (sample == "ZZ"):
            hists["VV"].append(mc_errors[sample])
        elif "TTT" in sample:
            hists["tt"].append(mc_errors[sample])
        elif "ST" in sample:
            hists["SingleTop"].append(mc_errors[sample])
        elif ("VBFH" in sample) or ("GluGluH" in sample):
            hists["Higgs"].append(mc_errors[sample])
    
    # accumulate mc errors by process
    for sample in hists:
        hists[sample] = processor.accumulate(hists[sample])
        
    # accumulate mc errors by variable
    total_bkg_error = processor.accumulate([hists[sample] for sample in hists])

    return total_bkg_error