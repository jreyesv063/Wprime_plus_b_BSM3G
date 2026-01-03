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
from wprime_plus_b.processors.btag_efficiency_processor import BTagEfficiencyProcessor
from wprime_plus_b.processors.ctag_efficiency_processor import CTagEfficiencyProcessor



def main(args):

    args = vars(args)
    # ==================================================
    #        Define processors and executors
    # ==================================================
    processors = {
        "ztoll": ZToLLProcessor,
        "zplusc": ZplusCProcessor,
        "btag_eff": BTagEfficiencyProcessor,
        "ctag_eff": CTagEfficiencyProcessor,
        "top_tagger": TopTaggerProccessor,
        "signal": SignalProccessor,
        "wjets": WjetsProccessor,
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
    print(">>> Llamando get_filesets")
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
        if "metadata" in out[sample]:
            output_metadata = out[sample]["metadata"]

            # save top_tagger metadata
            if args["processor"] in ["top_tagger", "signal", "qcd_hadronic_closure", "wplusjets"]:
                # Define the common keys and corresponding suffixes
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
            if args["processor"] in ["ztoll", "zplusc", "top_tagger", "signal", "wjets", "qcd_hadronic_closure", "wplusjets"]:

                # Step 1: create `main` with baseline info
                metadata.update({
                    "main": {
                        "nominal": {k: str(v) for k, v in output_metadata.get("cutflow", {}).items()},
                        "raw":     {k: str(v) for k, v in output_metadata.get("cutflow_raw", {}).items()},
                        "raw_initial_nevents": float(output_metadata["raw_initial_nevents"]),
                        "raw_final_nevents": float(output_metadata.get("raw_final_nevents", 0.)),
                        "sumw": float(output_metadata["sumw"]),
                        "weighted_final_nevents": float(output_metadata.get("weighted_final_nevents", 0.)),
                        "weight_statistics": {k: str(v) for k, v in output_metadata.get("weight_statistics", {}).items()},
                    }
                })

                # Step 2: add extra fields only for non-data samples
                if args["sample"] not in ["MET", "SingleMuon", "SingleElectron", "Tau"]:
                    metadata["main"].update({
                        "sumw_no_object_weights": float(output_metadata["sumw_no_object_weights"]),
                        "sumw_POG": float(output_metadata["sumw_POG"]),
                        "sumw_POG_plus_no_trigger": float(output_metadata["sumw_POG_plus_no_trigger"]),
                    })


            # ============================================================
            #         Save event selection criteria
            # ============================================================
            # Load event selection criteria
            with open(f"wprime_plus_b/selections/{args['processor']}/event_selection_criteria.yaml") as f:
                criteria = yaml.safe_load(f)

            # ------------------------------------------------------------
            #          Save top tagger selectios on metadata
            # ------------------------------------------------------------
            if args["processor"] in ["top_tagger", "signal", "wplusjets", "qcd_hadronic_closure", "ztoll", "zplusc"]:  
             
                selections = {
                    "electron_selection":  criteria["electron"][args["lepton_flavor"]],
                    "muon_selection": criteria["muon"][args["lepton_flavor"]],
                    "tau_selection": criteria["tau"][args["lepton_flavor"]],
                    "bjet_selection": criteria["bjet"][args["lepton_flavor"]],
                    "cross_cleaning_selection": criteria["cross_cleaning"][args["lepton_flavor"]],
                    "trigger_selection": criteria["trigger"][args["lepton_flavor"]],
                }

                if args["processor"] not in ["ztoll", "zplusc"]:
                    selections["jet_selection"] = criteria["jet"][args["lepton_flavor"]]                    
                    selections["fatjet_selection"] = criteria["fatjet"][args["lepton_flavor"]]
                    selections["wjet_selection"] = criteria["wjet"][args["lepton_flavor"]]
                    selections["top_tagger_cases"] = criteria["top_tagger"][args["lepton_flavor"]]
                    selections["met_selection"] =  criteria["met"][args["lepton_flavor"]]

                metadata.update({"selections": selections})

                # Remove duplicates while preserving the original order
                triggers = list(dict.fromkeys(output_metadata["Triggers"]))


                # Update the metadata dictionary with the cleaned triggers list as a string
                metadata.update({"Triggers": str(triggers)})


                if "Triggers_eff" in output_metadata:
                    triggers_eff = list(dict.fromkeys(output_metadata["Triggers_eff"]))
                    metadata.update({"Triggers_eff": str(triggers_eff)})
                else:
                    metadata.update({"Triggers_eff": "Not activated"})


                if args["run_systematics"] == "true" and args["sample"] not in ["MET", "SingleMuon", "SingleElectron", "Tau"]:

                    syst_var_object = [
                        "muon_Rochester_up", "muon_Rochester_down",
                        "tau_TES_up", "tau_TES_down",
                        "jet_JES_up", "jet_JES_up",
                        "jet_JER_up", "jet_JER_up",
                        "fatjet_JES_up", "fatjet_JES_up",
                        "fatjet_JER_up", "fatjet_JER_up",     
                    ]

                    has_fatjets = output_metadata.get("Are there Fatjets?", False)
                    # Si no hay fatjets, eliminar sistemáticas relacionadas con fatjets
                    if not has_fatjets:
                        syst_var_object = [s for s in syst_var_object if "fatjet" not in s.lower()]

                    cutflow_syst = {}

                    for syst in syst_var_object:
                        cutflow_syst[syst] = {
                            "nominal":{k: str(v) for k, v in output_metadata.get(f"cutflow_({syst})", {}).items()}, 
                            "raw":{k: str(v) for k, v in output_metadata.get(f"cutflow_({syst})_raw", {}).items()}, 
                        }
                        metadata["main"]["systematic_variations"] = cutflow_syst
                    

                if args["qcd_data_driven"] == "true":

                    cutflow_BCD = {}
                    values_BCD = {}
                    cr_data_driven = ["cr_b", "cr_c", "cr_d"]

                    for cr in ["cr_b", "cr_c", "cr_d"]:
                        # Save cutflow
                        cutflow_BCD[cr] = {
                            "nominal": {k: str(v) for k, v in output_metadata.get(f"cutflow_{cr}", {}).items()}, 
                            "raw": {k: str(v) for k, v in output_metadata.get(f"cutflow_{cr}_raw", {}).items()}
                        }
                        # Save number of events
                        values_BCD[cr] = {
                            "raw_initial_nevents": float(output_metadata["raw_initial_nevents"]),
                            "raw_final_nevents": float(output_metadata.get(f"raw_final_nevents_{cr}", 0.)),
                            "sumw": float(output_metadata[f"sumw_{cr}"]),
                            "weighted_final_nevents": float(output_metadata.get(f"weighted_final_nevents_{cr}", 0.)),
                        }

                    metadata["BCD"] = {
                        "cutflow": cutflow_BCD,
                        "values":  values_BCD,
                    }


                    selections_BCD = {
                        "QCD_data_driven (CR B)":  criteria["data_driven_qcd_estimation"]["cr_b"][args["lepton_flavor"]],
                        "QCD_data_driven (CR C)":  criteria["data_driven_qcd_estimation"]["cr_c"][args["lepton_flavor"]],
                        "QCD_data_driven (CR D)":  criteria["data_driven_qcd_estimation"]["cr_d"][args["lepton_flavor"]]
                    }
                    metadata.update({"selections_BCD": selections_BCD})

                
                    if args["run_systematics"] == "true" and args["sample"] not in ["MET", "SingleMuon", "SingleElectron", "Tau"]:

                        syst_var_object = [
                            "muon_Rochester_up", "muon_Rochester_down",
                            "met_UNCLUSTERED_up", "met_UNCLUSTERED_down",
                            "tau_TES_up", "tau_TES_down",
                            "jet_JES_up", "jet_JES_up",
                            "jet_JER_up", "jet_JER_up",
                            "fatjet_JES_up", "fatjet_JES_up",
                            "fatjet_JER_up", "fatjet_JER_up",     
                        ]

                        has_fatjets = output_metadata.get("Are there Fatjets?", False)
                        # Si no hay fatjets, eliminar sistemáticas relacionadas con fatjets
                        if not has_fatjets:
                            syst_var_object = [s for s in syst_var_object if "fatjet" not in s.lower()]

                        cutflow_syst_cr = {}

                        for cr in ["cr_b", "cr_c", "cr_d"]:
                            cutflow_syst_cr[cr] = {}
                            for syst in syst_var_object:
                                cutflow_syst_cr[cr][syst] = {
                                    "nominal":{k: str(v) for k, v in output_metadata.get(f"cutflow_{cr}_({syst})", {}).items()}, 
                                    "raw":{k: str(v) for k, v in output_metadata.get(f"cutflow_{cr}_({syst})_raw", {}).items()}, 
                                }
                        
                        metadata["BCD"][f"systematic_variations"] =  cutflow_syst_cr                                  


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
        help="processor to be used {ttbar, ztoll, zplusc, qcd, trigger_eff, btag_eff, ctag_eff, signal, wjets, qcd_abcd} (default ttbar)",
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
        "--output_folder",
        dest="output_folder",
        type=str,
        default="false",
        help="Output folder (str)",
     )
    args = parser.parse_args()
    main(args)