import json
import time
import dask
import pickle
import argparse
import datetime
import numpy as np
import wprime_plus_b.utils
import importlib.resources
from pathlib import Path
from coffea import processor
from utils import get_filesets
from dask.distributed import Client
from humanfriendly import format_timespan
from distributed.diagnostics.plugin import UploadDirectory
from wprime_plus_b.utils import paths
from wprime_plus_b.processors.ttbar_analysis import TtbarAnalysis
from wprime_plus_b.processors.top_tagger_processor import TopTaggerProccessor
from wprime_plus_b.processors.wjets_processor import WjetsProccessor
from wprime_plus_b.processors.QCD_ABCD_processor import QCD_ABCD_Proccessor
from wprime_plus_b.processors.signal_processor import SignalProccessor
from wprime_plus_b.processors.ztoll_processor import ZToLLProcessor
from wprime_plus_b.processors.btag_efficiency_processor import BTagEfficiencyProcessor
from wprime_plus_b.selections.ttbar.electron_config import ttbar_electron_config
from wprime_plus_b.selections.ttbar.muon_config import ttbar_muon_config
from wprime_plus_b.selections.ttbar.tau_config import ttbar_tau_config
from wprime_plus_b.selections.ttbar.bjet_config import ttbar_bjet_config
from wprime_plus_b.selections.qcd.config import (
    qcd_electron_selection,
    qcd_muon_selection,
    qcd_jet_selection,
    qcd_tau_selection
)

# Top tagger configs
from wprime_plus_b.selections.top_tagger.bjet_config import top_tagger_bjet_selection
from wprime_plus_b.selections.top_tagger.cases_top_tagger_config import top_tagger_cases_selection, top_tagger_mW_mTop_Njets_selection
from wprime_plus_b.selections.top_tagger.electron_config import top_tagger_electron_selection
from wprime_plus_b.selections.top_tagger.fatjet_config import top_tagger_fatjet_selection
from wprime_plus_b.selections.top_tagger.general_config import top_tagger_cross_cleaning_selection, top_tagger_trigger_selection
from wprime_plus_b.selections.top_tagger.jet_config import top_tagger_jet_selection
from wprime_plus_b.selections.top_tagger.met_config import top_tagger_met_selection
from wprime_plus_b.selections.top_tagger.muon_config import top_tagger_muon_selection
from wprime_plus_b.selections.top_tagger.tau_config import top_tagger_tau_selection
from wprime_plus_b.selections.top_tagger.wjet_config import top_tagger_wjet_selection


# Signal configs
from wprime_plus_b.selections.signal.bjet_config import signal_bjet_selection
from wprime_plus_b.selections.signal.cases_top_tagger_config import signal_cases_selection
from wprime_plus_b.selections.signal.electron_config import signal_electron_selection
from wprime_plus_b.selections.signal.fatjet_config import signal_fatjet_selection
from wprime_plus_b.selections.signal.general_config import signal_cross_cleaning_selection, signal_trigger_selection
from wprime_plus_b.selections.signal.jet_config import signal_jet_selection
from wprime_plus_b.selections.signal.met_config import signal_met_selection
from wprime_plus_b.selections.signal.muon_config import signal_muon_selection
from wprime_plus_b.selections.signal.tau_config import signal_tau_selection
from wprime_plus_b.selections.signal.wjet_config import signal_wjet_selection



# WJets configs
from wprime_plus_b.selections.wjets.bjet_config import wjet_bjet_selection
from wprime_plus_b.selections.wjets.electron_config import wjet_electron_selection
from wprime_plus_b.selections.wjets.general_config import wjet_cross_cleaning_selection, wjet_trigger_selection
from wprime_plus_b.selections.wjets.jet_config import wjet_jet_selection
from wprime_plus_b.selections.wjets.leading_jet_config import wjet_leading_jet_selection
from wprime_plus_b.selections.wjets.met_config import wjet_met_selection
from wprime_plus_b.selections.wjets.muon_config import wjet_muon_selection
from wprime_plus_b.selections.wjets.tau_config import wjet_tau_selection


# QCD_ABC configs
from wprime_plus_b.selections.QCD_ABCD.bjet_config import QCD_ABCD_bjet_selection
from wprime_plus_b.selections.QCD_ABCD.electron_config import QCD_ABCD_electron_selection
from wprime_plus_b.selections.QCD_ABCD.general_config import QCD_ABCD_cross_cleaning_selection, QCD_ABCD_trigger_selection
from wprime_plus_b.selections.QCD_ABCD.mt_config import QCD_ABCD_mt_selection 
from wprime_plus_b.selections.QCD_ABCD.met_config import QCD_ABCD_met_selection
from wprime_plus_b.selections.QCD_ABCD.muon_config import QCD_ABCD_muon_selection
from wprime_plus_b.selections.QCD_ABCD.tau_config import QCD_ABCD_tau_selection

# Ztoll configs
from wprime_plus_b.selections.ztoll.bjet_config import ztoll_bjet_selection
from wprime_plus_b.selections.ztoll.electron_config import ztoll_electron_selection
from wprime_plus_b.selections.ztoll.general_config import ztoll_cross_cleaning_selection, ztoll_trigger_selection
from wprime_plus_b.selections.ztoll.jet_config import ztoll_jet_selection
from wprime_plus_b.selections.ztoll.leading_jet_config import ztoll_leading_jet_selection
from wprime_plus_b.selections.ztoll.met_config import ztoll_met_selection
from wprime_plus_b.selections.ztoll.muon_config import ztoll_muon_selection
from wprime_plus_b.selections.ztoll.tau_config import ztoll_tau_selection
from wprime_plus_b.selections.ztoll.Z_config import ztoll_charges_selection, ztoll_mrec_ll_selection



def main(args):
    args = vars(args)
    # define processors and executors
    processors = {
        "ttbar": TtbarAnalysis,
        "ztoll": ZToLLProcessor,
        #"qcd": QcdAnalysis,
        "btag_eff": BTagEfficiencyProcessor,
        #"trigger_eff": TriggerEfficiencyProcessor,
        "top_tagger": TopTaggerProccessor,
        "signal": SignalProccessor,
        "wjets": WjetsProccessor,
        "qcd_abcd": QCD_ABCD_Proccessor,
    }
    processor_args = [
        "year",
        "channel",
        "lepton_flavor",
        "output_type",
        "syst",
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
        
    # get .json filesets for sample
    filesets = get_filesets(
        sample=args["sample"],
        year=args["year"],
        facility=args["facility"],
    )
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
             fileset[sample] = [f"root://xrootd-vanderbilt.sites.opensciencegrid.org:1094/" + file for file in root_file]       
        elif sample.startswith("SignalMuon") or sample.startswith("SignalElectron"):
            fileset[sample] = [f"root://eoscms.cern.ch//eos/cms/" + file for file in root_file]
        elif args["facility"] == "coffea-casa":
            fileset[sample] = [f"root://xcache/" + file for file in root_file]
        else:
            fileset[sample] = root_file

        # run processor
        t0 = time.monotonic()
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
        if "metadata" in out[sample]:
            output_metadata = out[sample]["metadata"]
            # save number of raw initial events
            metadata.update({"raw_initial_nevents": float(output_metadata["raw_initial_nevents"])})
            # save number of weighted initial events
            if args["processor"] == "qcd":
                if args["channel"] != "all":
                    metadata.update({"sumw": float(output_metadata[args["channel"]]["sumw"])})
                else:
                    sumws = {}
                    for r in ["A", "B", "C", "D"]:
                        sumws[r] = float(output_metadata[r]["sumw"])
                    metadata.update({"sumw": sumws})
            else:
                metadata.update({"sumw": float(output_metadata["sumw"])})
            # save qcd metadata
            if args["processor"] in ["qcd"]:
                metadata.update({"nevents": {}})
                region = args["channel"]
                if region != "all":
                    metadata["nevents"].update({region: {}})
                    metadata["nevents"][region]["raw_final_nevents"] = str(
                        output_metadata[region]["raw_final_nevents"]
                    )
                    metadata["nevents"][region]["weighted_final_nevents"] = str(
                        output_metadata[region]["weighted_final_nevents"]
                    )
                elif region == "all":
                    for r in ["A", "B", "C", "D"]:
                        metadata["nevents"].update({r: {}})
                        metadata["nevents"][r]["raw_final_nevents"] = str(
                            output_metadata[r]["raw_final_nevents"]
                        )
                        metadata["nevents"][r]["weighted_final_nevents"] = str(
                            output_metadata[r]["weighted_final_nevents"]
                        )
            # save top_tagger metadata
            if args["processor"] in ["top_tagger", "signal"]:
                # Define las claves comunes y los sufijos correspondientes
                all_keys = [
                    "one_jet_unresolve", "two_jets_unresolve", "two_jets_partially_resolve", 
                    "three_jets_partially_resolve", "three_jets_resolve", "four_jets_resolve", 
                    "N_jets_resolve", "N_bjets_resolve", "one_jet_unresolve_gen", 
                    "two_jets_unresolve_gen", "two_jets_partially_resolve_gen", 
                    "three_jets_partially_resolve_gen", "three_jets_resolve_gen"
                ]

                suffixes = ["triggered_raw", "triggered_nevents", "raw", "nevents"]

                # Función para construir un diccionario de entradas
                def build_entries_dict(output_metadata, suffix, all_keys):
                    entries = {}
                    for key in all_keys:
                        full_key = f"{key}_{suffix}"
                        if full_key in output_metadata:
                            entries[key] = float(output_metadata[full_key])
                    return entries

                # Generar los diccionarios de entradas dinámicamente
                top_tagger_triggered_raw_entries = build_entries_dict(output_metadata, "triggered_raw", all_keys)
                top_tagger_triggered_raw_entries["total"] = float(output_metadata.get("Total_triggered_raw", 0.0))

                top_tagger_triggered_nevents_entries = build_entries_dict(output_metadata, "triggered_nevents", all_keys)
                top_tagger_triggered_nevents_entries["total"] = float(output_metadata.get("Total_triggered_nevents", 0.0))

                top_tagger_raw_entries = build_entries_dict(output_metadata, "raw", all_keys)
                top_tagger_raw_entries["total"] = float(output_metadata.get("Total_raw", 0.0))

                top_tagger_entries = build_entries_dict(output_metadata, "nevents", all_keys)
                top_tagger_entries["total"] = float(output_metadata.get("Total_nevents", 0.0))

                # Actualizar el diccionario de metadata
                metadata.update({
                    "top_tagger_triggered_raw": top_tagger_triggered_raw_entries,
                    "top_tagger_triggered_nevents": top_tagger_triggered_nevents_entries,
                    "top_tagger_raw": top_tagger_raw_entries,
                    "top_tagger_nevents": top_tagger_entries
                })            
                        
            # save metadata
            if args["processor"] in ["ttbar", "ztoll", "top_tagger", "signal", "wjets", "qcd_abcd"]:
             
                # save raw and weighted number of events after selection
                if "raw_final_nevents" in output_metadata:
                    metadata.update(
                        {"raw_final_nevents": float(output_metadata["raw_final_nevents"])}
                    )
                    metadata.update(
                        {"weighted_final_nevents": float(output_metadata["weighted_final_nevents"])}
                    )
                else:
                    metadata.update(
                        {"raw_final_nevents": 0.}
                    )
                    metadata.update(
                        {"weighted_final_nevents": 0.}
                    )
                # save cutflow to metadata
                for cut_selection, nevents in output_metadata["cutflow"].items():
                    output_metadata["cutflow"][cut_selection] = str(nevents)
                metadata.update({"cutflow": output_metadata["cutflow"]})

                for weight, statistics in output_metadata["weight_statistics"].items():
                    output_metadata["weight_statistics"][weight] = str(statistics)
                metadata.update(
                    {"weight_statistics": output_metadata["weight_statistics"]}
                )
            # save selectios to metadata
            if args["processor"] == "ttbar": 
                selections = {
                    "electron_selection": ttbar_electron_config[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "muon_selection": ttbar_muon_config[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "jet_selection": ttbar_bjet_config[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "tau_selection": ttbar_tau_config[args["channel"]][
                        args["lepton_flavor"]
                    ]
                }
                metadata.update({"selections": selections})

            elif args["processor"] == "qcd":  
                region = args["channel"]
                if region != "all":
                    selections = {
                        "electron_selection": qcd_electron_selection[region][args["lepton_flavor"]],
                        "muon_selection": qcd_muon_selection[region][args["lepton_flavor"]],
                        "jet_selection": qcd_jet_selection[region][args["lepton_flavor"]],
                        "tau_selection": qcd_tau_selection[region][args["lepton_flavor"]],
                    }
                    metadata.update({"selections": selections})
                elif region == "all":
                    selections = {}
                    for r in ["A", "B", "C", "D"]:
                        selections[r] = {
                            "electron_selection": qcd_electron_selection[r][args["lepton_flavor"]],
                            "muon_selection": qcd_muon_selection[r][args["lepton_flavor"]],
                            "jet_selection": qcd_jet_selection[r][args["lepton_flavor"]],
                            "tau_selection": qcd_tau_selection[r][args["lepton_flavor"]],
                        }
                        metadata.update({"selections": selections})

             
            # save top tagger selectios to metadata
            elif args["processor"] in ["top_tagger"]:  
             
                selections = {
                    "electron_selection": top_tagger_electron_selection[
                        args["lepton_flavor"]
                        ],
                    "muon_selection": top_tagger_muon_selection[
                        args["lepton_flavor"]
                        ],
                    "tau_selection": top_tagger_tau_selection[
                        args["lepton_flavor"]
                        ],
                    "bjet_selection": top_tagger_bjet_selection[
                        args["lepton_flavor"]
                        ],
                    "jet_selection": top_tagger_jet_selection[
                        args["lepton_flavor"]
                        ],
                    "fatjet_selection": top_tagger_fatjet_selection[
                        args["lepton_flavor"]
                        ],
                    "wjet_selection": top_tagger_wjet_selection[
                        args["lepton_flavor"]
                        ],
                    "cross_cleaning_selection": top_tagger_cross_cleaning_selection[
                        args["lepton_flavor"]
                        ],
                    "Mass window for TOPs and Ws in NJets scenarios": top_tagger_mW_mTop_Njets_selection[
                        args["lepton_flavor"]
                        ],
                    "trigger_selection": top_tagger_trigger_selection[
                        args["lepton_flavor"]
                        ],
                    "met_selection": top_tagger_met_selection[
                        args["lepton_flavor"]
                        ],
                    "trigger_selection": top_tagger_trigger_selection[
                        args["lepton_flavor"]
                        ],
                    "top_tagger_cases": top_tagger_cases_selection[
                        args["lepton_flavor"]
                        ],
                }
                metadata.update({"selections": selections})

            # save top tagger selectios to signal
            elif args["processor"] in ["signal"]:  
             
                selections = {
                    "electron_selection": signal_electron_selection[
                        args["lepton_flavor"]
                        ],
                    "muon_selection": signal_muon_selection[
                        args["lepton_flavor"]
                        ],
                    "tau_selection": signal_tau_selection[
                        args["lepton_flavor"]
                        ],
                    "bjet_selection": signal_bjet_selection[
                        args["lepton_flavor"]
                        ],
                    "jet_selection": signal_jet_selection[
                        args["lepton_flavor"]
                        ],
                    "fatjet_selection": signal_fatjet_selection[
                        args["lepton_flavor"]
                        ],
                    "wjet_selection": signal_wjet_selection[
                        args["lepton_flavor"]
                        ],
                    "cross_cleaning_selection": signal_cross_cleaning_selection[
                        args["lepton_flavor"]
                        ],
                    "trigger_selection": signal_trigger_selection[
                        args["lepton_flavor"]
                        ],
                    "met_selection": signal_met_selection[
                        args["lepton_flavor"]
                        ],
                    "trigger_selection": signal_trigger_selection[
                        args["lepton_flavor"]
                        ],
                    "top_tagger_cases": signal_cases_selection[
                        args["lepton_flavor"]
                        ],
                }
                metadata.update({"selections": selections})


            # save wjets selectios to metadata
            elif args["processor"] in ["wjets"]:  
             
                selections = {
                    "electron_selection": wjet_electron_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "muon_selection": wjet_muon_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "tau_selection": wjet_tau_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "jet_selection": wjet_jet_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "leading_jet_selection": wjet_leading_jet_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "bjet_selection": wjet_bjet_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "met_selection": wjet_met_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "cross_cleaning_selection": wjet_cross_cleaning_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "trigger_selection":wjet_trigger_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                }
                metadata.update({"selections": selections})

           # save qcd_abcd selectios to metadata
            elif args["processor"] in ["qcd_abcd"]:  
             
                selections = {
                    "electron_selection": QCD_ABCD_electron_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "muon_selection": QCD_ABCD_muon_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "tau_selection": QCD_ABCD_tau_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "bjet_selection": QCD_ABCD_bjet_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "met_selection": QCD_ABCD_met_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "mt_selection": QCD_ABCD_mt_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "cross_cleaning_selection": QCD_ABCD_cross_cleaning_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "trigger_selection":QCD_ABCD_trigger_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                }
                metadata.update({"selections": selections})

            # save ztoll selectios to metadata
            elif args["processor"] in ["ztoll"]:  
            
                selections = {
                    "electron_selection": ztoll_electron_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "muon_selection": ztoll_muon_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "tau_selection": ztoll_tau_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "met_selection": ztoll_met_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "Charge_dilepton": ztoll_charges_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "Dilepton_mass_rec": ztoll_mrec_ll_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                }

                if args["channel"] in ["ll_ISR"]:
                    selections.update({

                    "jet_selection": ztoll_jet_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "leading_jet_selection": ztoll_leading_jet_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "bjet_selection": ztoll_bjet_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                })
                        
                metadata.update({"selections": selections})


        with importlib.resources.path(
                    "wprime_plus_b.data", "triggers.json"
        ) as path:
            with open(path, "r") as handle:
                triggers = json.load(handle)[args["year"] ] 


        if args["processor"] in ["top_tagger", "signal", "wjets", "qcd_abcd"]:
         
            if args["lepton_flavor"] in ["tau"]:
                trigger_option =  top_tagger_trigger_selection[args["lepton_flavor"]]["trigger"]
                trigger_name = triggers[trigger_option]
                metadata.update({"Trigger name": trigger_name})


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
        help="processor to be used {ttbar, ztoll, qcd, trigger_eff, btag_eff, signal, wjets, qcd_abcd} (default ttbar)",
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
    args = parser.parse_args()
    main(args)