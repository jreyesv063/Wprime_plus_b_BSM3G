import json
import sys
import time
import dask
import yaml
import pickle
import argparse
import datetime
import numpy as np
from pathlib import Path
import wprime_plus_b.utils
import importlib.resources
from coffea import processor
from utils import get_filesets
from dask.distributed import Client
from wprime_plus_b.utils import paths
from humanfriendly import format_timespan
from distributed.diagnostics.plugin import UploadDirectory

# Processors
from wprime_plus_b.processors.ztoll_processor import ZToLLProcessor
from wprime_plus_b.processors.wjets_processor import WjetsProccessor
from wprime_plus_b.processors.signal_processor import SignalProccessor
from wprime_plus_b.processors.zplusc_processor import ZplusCProcessor
from wprime_plus_b.processors.wplusjets_processor import WplusJetsProcessor
from wprime_plus_b.processors.top_tagger_processor import TopTaggerProccessor
from wprime_plus_b.processors.qcd_closure_processor import QCD_closure_Proccessor
#from wprime_plus_b.processors.btag_efficiency_processor import BTagEfficiencyProcessor
#from wprime_plus_b.processors.ctag_efficiency_processor import CTagEfficiencyProcessor



def main(args):

    args = vars(args)
    # ==================================================
    #        Define processors and executors
    # ==================================================
    processors = {
        "ztoll": ZToLLProcessor,
        "wjets": WjetsProccessor,
        "zplusc": ZplusCProcessor,
        #"btag_eff": BTagEfficiencyProcessor,
        #"ctag_eff": CTagEfficiencyProcessor,
        "top_tagger": TopTaggerProccessor,
        "signal": SignalProccessor,
        "qcd_hadronic_closure": QCD_closure_Proccessor,
        "wplusjets": WplusJetsProcessor,
    }
    processor_args = [
        "year",
        "processor",
        "channel",
        "lepton_flavor",
        "output_type",
        "syst",
        "run_systematics",
        "qcd_data_driven",
        "unblinded",
        "output_folder"
    ]
    processor_kwargs = {k: args[k] for k in processor_args if args[k]}
    executors = {
        "iterative": processor.iterative_executor,
        "futures": processor.futures_executor,
        "dask": processor.dask_executor,
    }
    executor_args = {
        "schema": processor.NanoAODSchema,
    }
    if args["executor"] == "futures":
        executor_args.update({"workers": args["workers"]})
    if args["executor"] == "dask":
        client = Client("tls://localhost:8786")
        #executor_args["client"] = client
        executor_args.update({"client": client})
        # upload local directory to dask workers
        try:
            client.register_worker_plugin(
                UploadDirectory(f"{Path.cwd()}", restart=True, update_path=True),
                nanny=True,
            )
            print(f"Uploaded {Path.cwd()} succesfully")
        except OSError:
            print("Failed to upload the directory")
        
    # ======================================================
    #           Load filesets 
    # ======================================================
    # get .json filesets for sample
    filesets = get_filesets(
        sample=args["sample"],
        year=args["year"],
        facility=args["facility"],
    )

    sample = args['sample']
    nsample = args.get('nsample')

    if nsample:
        fileset_name_nsample = f"{args['sample']}_{args['nsample']}"
    else:
        fileset_name_nsample = f"{args['sample']}"


    if fileset_name_nsample not in filesets:
        print(f"❌ Error :  No fileset {fileset_name_nsample} found in fileset/{args['year']}/{args['facility']}. Check the folder", file=sys.stderr)
        return
    else:
        print(f"✅ Fileset found: fileset/{args['year']}/{args['facility']}/{fileset_name_nsample}.json")


    for sample, fileset_path in filesets.items():

        if len(args["nsample"]) != 0:
            samples_keys = args["nsample"].split(",")
            if sample.split("_")[-1] not in samples_keys:
                continue
        print(f"Processing {sample}")
        fileset = {}
        with open(fileset_path, "r") as handle:
            data = json.load(handle)
        for root_file in data.values():
            if args["nfiles"] != -1:
                root_file = root_file[: args["nfiles"]]


        if sample.startswith("SignalTau"):
            fileset[sample] = [f"root://cmsxrootd.fnal.gov/" + file for file in root_file]
        elif sample.startswith("SignalMuon") or sample.startswith("SignalElectron"):
            fileset[sample] = [f"root://eoscms.cern.ch//eos/cms/" + file for file in root_file]
        elif args["facility"] == "coffea-casa":
            fileset[sample] = [f"root://xcache/" + file for file in root_file]
        else:
            fileset[sample] = root_file

        # run processor
        t0 = time.monotonic()
        print(processor_kwargs)
        out = processor.run_uproot_job(
            fileset,
            treename="Events",
            processor_instance=processors[args["processor"]](**processor_kwargs),
            executor=executors[args["executor"]],
            executor_args=executor_args,
        )
        exec_time = format_timespan(time.monotonic() - t0)

        # get metadata
        metadata = {"walltime": exec_time}
        metadata.update({"fileset": fileset[sample]})

        # Save cutflows and other metadata
        if "metadata" in out[sample]:
            # ============================================================
            #         Save cutflows
            # ============================================================  
            output_metadata = out[sample]["metadata"]
            clean_metadata = output_metadata.copy()

            # Lista de secciones donde puede aparecer weight_statistics
            sections = ["main", "cr_b", "cr_c", "cr_d"]

            for section in sections:
                if section in clean_metadata and "weight_statistics" in clean_metadata[section]:
                    ws = clean_metadata[section].pop("weight_statistics")

                    # Convertimos cada estadística a string
                    for weight, statistics in ws.items():
                        ws[weight] = str(statistics)

                    # Guardamos en metadata de forma segura
                    clean_metadata.setdefault(section, {})["weight_statistics"] = ws

            # Actualizamos el resto de metadata
            metadata.update(clean_metadata)



            # ============================================================
            #         Save event selection criteria
            # ============================================================
            # Load event selection criteria
            with open(f"wprime_plus_b/selection_criteria/{args['processor']}/event_selection_criteria.yaml") as f:
                criteria = yaml.safe_load(f)

            lepton_flavor = args["lepton_flavor"]
            selection_criteria = {}

            for channel, values in criteria.items():
                # Caso 1: Estructura estándar (muon, electron, tau, jet, cross_cleaning, etc.)
                if isinstance(values, dict) and lepton_flavor in values:
                    selection_criteria[channel] = values[lepton_flavor]
                
                # Caso 2: Estructuras anidadas (como data_driven_qcd_estimation)
                elif isinstance(values, dict):
                    sub_dict = {}
                    for sub_channel, sub_values in values.items():
                        if isinstance(sub_values, dict) and lepton_flavor in sub_values:
                            sub_dict[sub_channel] = sub_values[lepton_flavor]
                    
                    if sub_dict: # Solo lo guardamos si encontramos algo para ese sabor
                        selection_criteria[channel] = sub_dict

            metadata["selections"] = selection_criteria


        # save args to metadata
        args_dict = args.copy()
        metadata.update(args_dict)
        if "metadata" in out[sample]:
            del out[sample]["metadata"]
        # save output data and metadata
        with open(f"{args['output_path']}/metadata/{sample}_metadata.json", "w") as f:
            f.write(json.dumps(metadata))
        with open(f"{args['output_path']}/{sample}.pkl", "wb") as handle:
            pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--processor",
        dest="processor",
        type=str,
        default="",
        help="processor to be used {ttbar, ztoll, zplusc, trigger_eff, btag_eff, ctag_eff, signal, wjets} (default ttbar)",
    )
    parser.add_argument(
        "--channel",
        dest="channel",
        type=str,
        default="",
        help="channel to be processed",
    )
    parser.add_argument(
        "--lepton_flavor",
        dest="lepton_flavor",
        type=str,
        default="",
        help="lepton flavor to be processed {'mu', 'ele'}",
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
        default="",
        help="type of output {hist, array}",
    )
    parser.add_argument(
        "--syst",
        dest="syst",
        type=str,
        default="",
        help="systematic to apply {'nominal', 'jet', 'met', 'full'}",
    )
    parser.add_argument(
        "--facility",
        dest="facility",
        type=str,
        default="",
        help="facility to launch jobs {coffea-casa, lxplus}",
    )
    parser.add_argument(
        "--tag",
        dest="tag",
        type=str,
        default="",
        help="tag to reference output files directory",
    )
    parser.add_argument(
        "--output_path",
        dest="output_path",
        type=str,
        default="",
        help="output path directory",
    )
    parser.add_argument(
        "--run_systematics",
        dest="run_systematics",
        type=str,
        default="false",
        help="Run systematics (true/false)",
     )

    parser.add_argument(
        "--qcd_data_driven",
        dest="qcd_data_driven",
        type=str,
        default="false",
        help="Run systematics (true/false)",
     )     

    parser.add_argument(
        "--unblinded",
        dest="unblinded",
        type=str,
        default="false",
        help="Use data in SR",
     )       

    parser.add_argument(
        "--global_redirector",
        dest="global_redirector",
        type=str,
        default="false",
        help="Use global redirector",
    )
    
    parser.add_argument(
        "--output_folder",
        dest="output_folder",
        type=str,
        default="false",
        help="Output folder (str)",
     )
    args = parser.parse_args()
    main(args)