import os
import argparse
from utils import build_filesets, get_command, run_checker, manage_processor_args, build_output_directories


def main(args):
    args = manage_processor_args(vars(args))
    run_checker(args)
    # add facility and output path to args
    args["facility"] = "coffea-casa"
    args["output_path"] = build_output_directories(args)
    # build filesets
    build_filesets(args)
    # run command
    cmd = get_command(args)
    os.system(cmd)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--processor",
        dest="processor",
        type=str,
        default="",
        help="processor to be used {ttbar, ztoll, qcd, trigger_eff, btag_eff, top_tagger, signal, wjets} (default ttbar)",
    )
    parser.add_argument(
        "--channel",
        dest="channel",
        type=str,
        default="",
        help="channel to be processed {'2b1l', '1b1e1mu', '1b1l', 'll', 'll_ISR'}",
    )
    parser.add_argument(
        "--lepton_flavor",
        dest="lepton_flavor",
        type=str,
        default="",
        help="lepton flavor to be processed {'mu', 'ele', 'tau'}",
    )
    parser.add_argument(
        "--sample",
        dest="sample",
        type=str,
        default="",
        help="sample key to be processed",
    )
    parser.add_argument(
        "--year",
        dest="year",
        type=str,
        default="",
        help="year of the data {2016, 2017, 2018} (default 2017)",
    )
    parser.add_argument(
        "--yearmod",
        dest="yearmod",
        type=str,
        default="",
        help="year modifier {'', 'APV'} (default '')",
    )
    parser.add_argument(
        "--executor",
        dest="executor",
        type=str,
        default="",
        help="executor to be used {iterative, futures, dask} (default iterative)",
    )
    parser.add_argument(
        "--workers",
        dest="workers",
        type=int,
        default=4,
        help="number of workers to use with futures executor (default 4)",
    )
    parser.add_argument(
        "--nfiles",
        dest="nfiles",
        type=int,
        default=1,
        help="number of .root files to be processed by sample. To run all files use -1 (default 1)",
    )
    parser.add_argument(
        "--nsample",
        dest="nsample",
        type=str,
        default="",
        help="partitions to run (--nsample 1,2,3 will only run partitions 1,2 and 3)",
    )
    parser.add_argument(
        "--chunksize",
        dest="chunksize",
        type=int,
        default=50000,
        help="number of chunks to process",
    )
    parser.add_argument(
        "--output_type",
        dest="output_type",
        type=str,
        default="hist",
        help="type of output {hist, array}",
    )
    parser.add_argument(
        "--syst",
        dest="syst",
        type=str,
        default="nominal",
        help="systematic to apply {'nominal', 'jet', 'met', 'full'}",
    )
    parser.add_argument(
        "--tag",
        dest="tag",
        type=str,
        default="",
        help="tag to reference output files directory",
    )
    args = parser.parse_args()
    main(args)
    