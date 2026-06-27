import os
import math
import json
import glob
import yaml
from tqdm import tqdm
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
    #processor_config = load_processor_config(config_name=processor_config_name)

    processor_config = load_processor_config(
        processor = args["processor"],
        channel = args["channel"],
        lepton_flavor = args["lepton_flavor"]
    )

    # get processor output path
    processor_output_path = paths.processor_path(
        processor_name=processor_config.name,
        processor_lepton_flavour=processor_config.lepton_flavor,
        processor_channel=processor_config.channel,
        processor_output=os.path.join(
            os.environ.get("ANALYSIS_PATH", ""),
            args["output_folder"]
        ),
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
    Build filesets partitions for a specific facility by combining signal and background datasets.
    
    Args:
        args (dict): Configuration dictionary containing:
            - year: Data year (e.g., '2016', '2017', '2018')
            - facility: Computing facility ('lxplus' or others)
            - sample: Sample name (optional, for legacy support)
            - run_systematics: Boolean flag for systematics processing
    """
    main_dir = Path.cwd()
    fileset_path = Path(f"{main_dir}/wprime_plus_b/fileset")

    # Determine JSON files to load based on facility and signal/background
    json_files = [
        f"{fileset_path}/signal_{args['year']}.json",  # Signal datasets
        f"{fileset_path}/fileset_{args['year']}_UL_NANO_lxplus.json" 
        if args['facility'] == "lxplus" 
        else f"{fileset_path}/fileset_{args['year']}_UL_NANO.json"  # Background datasets
    ]

    # Combine datasets from all JSON files
    combined_datasets = {}
    for json_file in json_files:
        if Path(json_file).exists():
            with open(json_file, "r") as f:
                try:
                    data = json.load(f)

                    # ------------------------------------------------------------
                    # Select global redirector ig the variable is set to true
                    # ------------------------------------------------------------
                    if args["global_redirector"] == "true":
                        new_data = {}
                        for sample, paths in data.items():
                            clean_paths = []
                            for p in paths:
                                if "/store" in p:
                                    # Obtain the path after "/store" 
                                    lfn = p.split("/store")[1]
                                    
                                    if sample.startswith("Signal"):
                                        #clean_paths.append(f"root://cmsxrootd.fnal.gov//store/{lfn}")
                                        clean_paths.append(f"/store/{lfn}")
                                    else:
                                        clean_paths.append(f"root://cms-xrd-global.cern.ch//store/{lfn}")

                                else:
                                    clean_paths.append(p)

                            new_data[sample] = clean_paths
                        data = new_data
                    # ------------------------------------------------------------

                    combined_datasets.update(data)
                except json.JSONDecodeError as e:
                    print(f"Error loading {json_file}: {e}")
                    continue

    # Prepare output directory (clean if exists)
    output_directory = Path(f"{fileset_path}/{args['year']}/{args['facility']}")
    
    if output_directory.exists():
        # Remove existing files with progress bar
        existing_files = list(output_directory.glob("*"))
        for file in tqdm(existing_files, desc="Deleting old files", unit="file"):
            if file.is_file():
                try:
                    file.unlink()
                except Exception as e:
                    print(f"Error deleting {file}: {e}")
    else:
        output_directory.mkdir(parents=True, exist_ok=True)

    # First pass: Count total files that will be generated
    total_files = 0
    for sample in combined_datasets:
        dataset_config = load_dataset_config(
            config_name=sample,
            object_syst=args.get("run_systematics", False)
        )
        # Count 1 file if no split, or N files if split
        total_files += 1 if dataset_config.nsplit == 1 else dataset_config.nsplit

    # Second pass: Process samples with accurate progress tracking
    with tqdm(total=total_files, desc="Generating JSON files", unit="file") as pbar:
        for sample in combined_datasets:
            try:
                dataset_config = load_dataset_config(
                    config_name=sample,
                    object_syst=args.get("run_systematics", False)
                )
                
                if dataset_config.nsplit == 1:
                    # Case: Single file per sample
                    output_path = output_directory / f"{sample}.json"
                    with open(output_path, "w") as f:
                        json.dump({sample: combined_datasets[sample]}, f, indent=4, sort_keys=True)
                    pbar.update(1)  # Update progress by 1 file
                else:
                    # Case: Split sample into multiple files
                    root_files_list = divide_list(combined_datasets[sample], dataset_config.nsplit)
                    for i in range(1, dataset_config.nsplit + 1):
                        key = f"{sample}_{i}"
                        output_path = output_directory / f"{key}.json"
                        with open(output_path, "w") as f:
                            json.dump({key: root_files_list[i-1]}, f, indent=4, sort_keys=True)
                        pbar.update(1)  # Update progress for each split file
                        
            except Exception as e:
                tqdm.write(f"Error processing {sample}: {e}")
                continue


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
        "ctag_eff": ["lepton_flavor", "channel", "syst"],
        "trigger_eff": ["channel", "syst"],
    }
    processor = args.get("processor")
    if processor in processor_args_mapping:
        for arg in processor_args_mapping[processor]:
            args[arg] = None
    return args


def update_nsplit(year: str) -> None:
    """
    Updates the `nsplit` value in the `datasets_configs_systematics.yaml` file
    based on the number of entries in the corresponding JSON file for the given year.

    Args:
        year (str): The year used to determine the JSON file (e.g., "2016", "2016APV", "2017", "2018").
    """
    # Determine the corresponding JSON file
    json_filename = f"fileset_{year}_UL_NANO_lxplus.json"
    json_path = Path(f"wprime_plus_b/fileset/{json_filename}")
    
    # Check if the JSON file exists
    if not json_path.exists():
        raise FileNotFoundError(f"The JSON file {json_filename} does not exist at {json_path}")

    # Read the JSON file
    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)

    # Read the YAML file
    yaml_path = Path("wprime_plus_b/configs/dataset/datasets_configs_systematics.yaml")
    if not yaml_path.exists():
        raise FileNotFoundError(f"The YAML file {yaml_path} does not exist.")

    with open(yaml_path, "r") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
   
    # Update the `nsplit` value in the YAML
    for dataset, config in yaml_data.items():
        if dataset in json_data:
            # Count the number of entries in the JSON for this dataset
            nsplit_count = len(json_data[dataset])
            # Update the `nsplit` value in the YAML
            config["nsplit"] =  max(1, math.ceil(len(json_data[dataset]) / 2))

    # Save the changes to the YAML file
    with open(yaml_path, "w") as yaml_file:
        yaml.safe_dump(yaml_data, yaml_file)

    print(f"YAML file successfully updated: {yaml_path}")



def run_checker(args: dict) -> None:
    # check processor
    available_processors = ["ttbar", "ztoll", "zplusc", "qcd", "btag_eff", "ctag_eff", "trigger_eff", "top_tagger", "signal", "wjets", "qcd_abcd", "qcd_hadronic", "qcd_hadronic_closure", "wplusjets"]
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

    object_systematic_variation = (args["run_systematics"].lower() == "true")

    # check sample
    #update_nsplit(args["year"])
    #configs_file= "datasets_configs.yaml" if object_systematic_variation else "datasets_configs.yaml"
    configs_file= "datasets_configs_systematics.yaml" if object_systematic_variation else "datasets_configs.yaml"
    configs_path = f"{Path.cwd()}/wprime_plus_b/configs/dataset/{configs_file}"


    with open(configs_path, "r") as stream:
        configs = yaml.safe_load(stream)
    available_samples = list(configs.keys())
    if args["sample"] not in available_samples:
        raise ValueError(
            f"Incorrect sample. Available samples are: {available_samples}"
        )
    # check nsample
    dataset_config = load_dataset_config(config_name=args["sample"], object_syst = args["run_systematics"])
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