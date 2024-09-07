import os
import json
import argparse
from pathlib import Path
from report import build_report
from processor_utils import group_outputs, accumulate_outputs


def main(args):
    # make output directory
    output_path = Path(f"./{args.tag}")
    if not output_path.exists():
        output_path.mkdir(parents=True)
        
    # load cross sections
    main_path = Path.cwd().parent.parent
    with open(f"{main_path}/wprime_plus_b/data/DAS_xsec.json", "r") as f:
        xsecs = json.load(f)
        
    # group and accumulate outputs
    grouped_outputs = group_outputs(args.output_directory)
    accumulated_outputs = accumulate_outputs(grouped_outputs)

    # save report to a csv file
    report = build_report(accumulated_outputs, xsecs, args.lumi)
    report.to_csv(f"{output_path}/report.csv")

    # generate and save plots
    assert ("SingleElectron" in accumulated_outputs) or (
        "SingleMuon" in accumulated_outputs
    ), "¡No data found!"
    os.system(
        f"python plotter.py --output_directory {args.output_directory} --tag {args.tag} --interval {args.interval} --channel {args.channel} --lumi {args.lumi}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_directory",
        dest="output_directory",
        type=str,
        default="",
        help="path to the output directory",
    )
    parser.add_argument(
        "--tag",
        dest="tag",
        type=str,
        default="test",
        help="tag label of the output directory to save results",
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