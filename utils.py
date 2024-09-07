import os
import json
import glob
import yaml
from pathlib import Path
from collections import OrderedDict
from wprime_plus_b.utils import paths
from wprime_plus_b.utils.load_config import load_dataset_config, load_processor_config


def build_output_directories(args: dict) -> str:
    """builds output directories for data and metadata. Return output path"""
    # get processor config
    processor_config_name = "_".join(
        [i for i in [args["processor"], args["channel"], args["lepton_flavor"]] if i]
    )
    processor_config = load_processor_config(config_name=processor_config_name)
    # get processor output path
    processor_output_path = paths.processor_path(
        processor_name=processor_config.name,
        processor_lepton_flavour=processor_config.lepton_flavor,
        processor_channel=processor_config.channel,
        dataset_year=args["year"],
        mkdir=True,
    )
    return processor_output_path


def get_command(args: dict) -> str:
    """return command to submit jobs at coffea-casa or lxplus"""
    cmd = f"python submit.py"
    for arg in args:
        if args[arg]:
            cmd += f" --{arg} {args[arg]}"
    return cmd


def divide_list(lst: list, n: int) -> list:
    """Divide a list into n sublists"""
    size = len(lst) // n
    remainder = len(lst) % n
    result = []
    start = 0
    for i in range(n):
        if i < remainder:
            end = start + size + 1
        else:
            end = start + size
        result.append(lst[start:end])
        start = end
    return result


def build_filesets(args: dict) -> None:
    """
    build filesets partitions for an specific facility
    """
    main_dir = Path.cwd()
    fileset_path = Path(f"{main_dir}/wprime_plus_b/fileset")
    if args['sample'].startswith("Signal"):
        with open(f"{fileset_path}/signal_{args['year']}.json", "r") as f:
            datasets = json.load(f)
    else:
        with open(f"{fileset_path}/das_datasets.json", "r") as f:
            datasets = json.load(f)[f"{args['year']}_UL"]


    # make output filesets directory
    output_directory = Path(f"{fileset_path}/{args['year']}/{args['facility']}")
    if output_directory.exists():
        for file in output_directory.glob("*"):
            if file.is_file():
                file.unlink()
    else:
        output_directory.mkdir(parents=True)
    for sample in datasets:
        if args['sample'].startswith("Signal"):            
            json_file = f"{fileset_path}/signal_{args['year']}.json"
        elif args['facility'] == "lxplus":
            json_file = f"{fileset_path}/fileset_{args['year']}_UL_NANO_lxplus.json"
        else:
            json_file = f"{fileset_path}/fileset_{args['year']}_UL_NANO.json"
            
        with open(json_file, "r") as handle:
            data = json.load(handle)
        # split fileset and save filesets
        filesets = {}
        # load dataset config
        dataset_config = load_dataset_config(config_name=sample)
        if dataset_config.nsplit == 1:
            filesets[sample] = f"{output_directory}/{sample}.json"
            sample_data = {sample: data[sample]}
            with open(f"{output_directory}/{sample}.json", "w") as json_file:
                json.dump(sample_data, json_file, indent=4, sort_keys=True)
        else:
            root_files_list = divide_list(data[sample], dataset_config.nsplit)
            keys = ".".join(
                f"{sample}_{i}" for i in range(1, dataset_config.nsplit + 1)
            ).split(".")
            for key, value in zip(keys, root_files_list):
                sample_data = {}
                sample_data[key] = list(value)

                filesets[key] = f"{output_directory}/{key}.json"
                with open(f"{output_directory}/{key}.json", "w") as json_file:
                    json.dump(sample_data, json_file, indent=4, sort_keys=True)


def get_filesets(sample: str, year: str, facility: str) -> dict:
    """return a dictionary with sample names as keys and .json files as values"""
    main_dir = Path.cwd()
    fileset_path = Path(f"{main_dir}/wprime_plus_b/fileset/{year}/{facility}")
    file_list = glob.glob(f"{fileset_path}/*.json")
    filesets = {}
    for file in file_list:
        file_name = file.split("/")[-1].replace(".json", "")
        if file_name.startswith(sample):
            filesets[file_name] = file
    if len(filesets) != 1:
        # sort the dictionary keys based on the number after the "_" in ascending order
        sorted_keys = sorted(filesets.keys(), key=lambda x: int(x.split("_")[-1]))
        # create an ordered dictionary using the sorted keys
        ordered_filesets = OrderedDict((key, filesets[key]) for key in sorted_keys)
        return ordered_filesets
    return filesets


def manage_processor_args(args: dict) -> dict:
    processor_args_mapping = {
        "qcd": ["syst"],
        "btag_eff": ["lepton_flavor", "channel", "syst"],
        "trigger_eff": ["channel", "syst"],
    }
    processor = args.get("processor")
    if processor in processor_args_mapping:
        for arg in processor_args_mapping[processor]:
            args[arg] = None
    return args


def run_checker(args: dict) -> None:
    # check processor
    available_processors = ["ttbar", "ztoll", "qcd", "btag_eff", "trigger_eff", "top_tagger", "signal", "wjets", "qcd_abcd"] 
    if args["processor"] not in available_processors:
        raise ValueError(
            f"Incorrect processor. Available processors are: {available_processors}"
        )
    # check executor
    available_executors = ["iterative", "futures"]
    if args["executor"] not in available_executors:
        raise ValueError(
            f"Incorrect executor. Available executors are: {available_executors}"
        )
    # check years
    available_years = ["2016APV", "2016", "2017", "2018"]
    if args["year"] not in available_years:
        raise ValueError(f"Incorrect year. Available years are: {available_years}")
    
    # check output type
    available_output_types = ["hist", "array"]
    if args["output_type"] not in available_output_types:
        raise ValueError(
            f"Incorrect output_type. Available output_types are: {available_output_types}"
        )
    # check sample
    configs_path = f"{Path.cwd()}/wprime_plus_b/configs/dataset/datasets_configs.yaml"
    with open(configs_path, "r") as stream:
        configs = yaml.safe_load(stream)
    available_samples = list(configs.keys())
    if args["sample"] not in available_samples:
        raise ValueError(
            f"Incorrect sample. Available samples are: {available_samples}"
        )
    # check nsample
    dataset_config = load_dataset_config(config_name=args["sample"])
    available_nsamples = [""] + [str(i) for i in range(1, dataset_config.nsplit + 1)]
    nsamples = args["nsample"].split(",")
    for nsample in nsamples:
        if nsample not in available_nsamples:
            raise ValueError(
                f"Incorrect nsample. Available nsamples are: {available_nsamples}"
            )
    if args["processor"] == "ttbar":
        # check channel
        available_channels = ["2b1l", "1b1e1mu", "1b1l"]
        if args["channel"] not in available_channels:
            raise ValueError(
                f"Incorrect channel. Available channels are: {available_channels}"
            )
        # check lepton flavor
        available_lepton_flavors = ["ele", "mu"]
        if args["lepton_flavor"] not in available_lepton_flavors:
            raise ValueError(
                f"Incorrect lepton flavor. Available lepton flavors are: {available_lepton_flavors}"
            )
        # check Data sample
        if args["lepton_flavor"] == "mu":
            if args["sample"] == "SingleElectron":
                    raise ValueError(
                        "muon channel should be run with SingleElectron dataset"
                    )
        else:
            if args["sample"] == "SingleMuon":
                    raise ValueError(
                        "electron channel should be run with SingleElectron dataset"
                    )
        # check systematics
        if args["output_type"] == "hist":
            available_systs = ["nominal", "jet", "met", "tau", "rochester", "full"]
            if args["syst"] not in available_systs:
                raise ValueError(
                    f"Incorrect syst. Available systs are: {available_systs}"
                )
    if args["processor"] == "qcd":
        # check channel
        available_channels = ["A", "B", "C", "D", "all"]
        if args["channel"] not in available_channels:
            raise ValueError(
                f"Incorrect channel. Available channels are: {available_channels}"
            )
        if args["lepton_flavor"] != "mu":
            raise ValueError("Only muon channel is available")
        if args["output_type"] != "hist":
            raise ValueError("Only histograms are available")