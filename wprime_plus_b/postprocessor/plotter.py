import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from plotter_utils import plot_histogram
from histograms import ttbar_cr_histograms
from processor_utils import (
    group_outputs,
    accumulate_outputs,
    fill_histograms,
    get_lumiweights,
    scale_histograms,
    group_histograms,
    get_mc_error,
)

def main(args):
    np.seterr(divide="ignore", invalid="ignore")

    main_path = Path.cwd().parent.parent

    # group and accumulate output
    grouped_outputs = group_outputs(args.output_directory)
    accumulated_outputs = accumulate_outputs(grouped_outputs)

    assert ("SingleElectron" in accumulated_outputs) or (
        "SingleMuon" in accumulated_outputs
    ), "¡No data found!"

    # fill hist histograms with output arrays
    if "ttbar" in args.output_directory:
        hist_histograms = ttbar_cr_histograms
    else:
        pass
    histograms = fill_histograms(accumulated_outputs, hist_histograms)

    # scale histograms to lumi-xsec
    with open(f"{main_path}/wprime_plus_b/data/DAS_xsec.json", "r") as f:
        xsecs = json.load(f)
    lumi_weights = get_lumiweights(accumulated_outputs, xsecs, args.lumi)
    scaled_histograms = scale_histograms(histograms, lumi_weights)

    # group scale histograms by process
    grouped_histograms = group_histograms(scaled_histograms)

    # define mc_errors
    if args.interval == "poisson":
        mc_errors = None
    else:
        mc_errors = get_mc_error(
            accumulated_outputs, ttbar_cr_histograms, xsecs, args.lumi
        )
    # make output directory
    output_path = Path(f"./{args.tag}")
    if not output_path.exists():
        output_path.mkdir(parents=True)
    # plot histograms
    for sample in grouped_histograms:
        for kin in tqdm(grouped_histograms[sample]):
            for var in grouped_histograms[sample][kin].axes.name:
                plot_histogram(
                    histograms=grouped_histograms,
                    kin=kin,
                    var=var,
                    mc_errors=mc_errors,
                    channel=args.channel,
                    output_dir=output_path,
                )
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_directory",
        dest="output_directory",
        type=str,
        help="path to the output directory",
    )
    parser.add_argument(
        "--tag",
        dest="tag",
        type=str,
        default="test",
        help="tag label of the output directory to save plots",
    )
    parser.add_argument(
        "--interval",
        dest="interval",
        type=str,
        default="none",
        help="uncertainty interval {'poisson', 'none'}",
    )
    parser.add_argument(
        "--channel",
        dest="channel",
        type=str,
        default="mu",
        help="lepton channel {'ele', 'mu'}",
    )
    parser.add_argument(
        "--lumi",
        dest="lumi",
        type=float,
        default=41477.877399,
        help="luminosity",
    )
    args = parser.parse_args()
    main(args)