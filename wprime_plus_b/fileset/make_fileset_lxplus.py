import os
import json
from coffea.dataset_tools.dataset_query import DataDiscoveryCLI


ERAS = {
    "2016APV": ["B1", "B2", "C", "D", "E","F"],
    "2016": ["F", "G", "H"],
    "2017": ["B", "C", "D", "E", "F"],
    "2018": ["A", "B", "C", "D"],
}

SITES = {
    "2016APV": [
        "T3_US_FNALLPC",
        "T1_US_FNAL_Disk",
#        "T2_US_Vanderbilt", # 26/07/2024
        "T2_US_Purdue",
        "T2_US_Nebraska",
        "T2_DE_DESY",
 #       "T2_BE_IIHE",  # 26/07/2024
        "T2_CH_CERN",
        "T1_DE_KIT_Disk",
        "T2_DE_RWTH",

        "T1_FR_CCIN2P3_Tape",
 #       "T2_IT_Legnaro",  # 18/07/2024
 #       "T2_IT_Rome",     # 18/07/2024
        "T2_TW_NCHC",      
        "T3_FR_IPNL",
 #       "T3_IT_Trieste",  # 18/07/2024
        "T2_UK_London_IC",
 #       "T1_RU_JINR_Disk",

    ],
    "2016": [
        "T3_US_FNALLPC",
#        "T1_US_FNAL_Disk",  # 05/08/2024
        "T2_US_Vanderbilt",
        "T2_US_Purdue",
        "T2_US_Nebraska",
        "T2_DE_DESY",
#        "T2_BE_IIHE",  # 27/07/2024
        "T2_CH_CERN",
        "T1_DE_KIT_Disk",
#        "T2_DE_RWTH", # 05/08/2024: (.rwth-aachen.de)
        "T2_BE_UCL",
#        "T1_UK_RAL_Disk",
        "T1_FR_CCIN2P3_Disk",

        "T3_FR_IPNL",
#        "T3_IT_Trieste", # 27/07/2024
        "T1_FR_CCIN2P3_Disk",
        "T1_FR_CCIN2P3_Tape",
        "T3_CH_CERN_OpenData",
#        "T1_RU_JINR_Disk",
    ],
    "2017": [
        "T2_BE_UCL",
#        "T2_ES_CIEMAT", 
        "T3_FR_IPNL",
#        "T3_US_Baylor",
#        "T2_UK_London_IC",
#        "T2_US_Nebraska",  # 31/07/2024 root://xrootd-local.unl.edu:1094
        "T2_DE_DESY",  # 01/08/2024
        "T2_CH_CERN", 
        "T1_FR_CCIN2P3_Tape",  
#        "T2_EE_Estonia", # 14/07/2024 
#        "T3_CH_PSI", # 02/08/2024
#        "T3_KR_KISTI", # 04/07/2024
#        "T3_KR_UOS", # 01/08/2024
#        "T3_US_NotreDame", # crc.nd.edu 01/08/2024
#        "T1_RU_JINR_Disk", # 11/08/2024
       "T3_US_FNALLPC", # 04/07/2024
       "T1_US_FNAL_Disk",  # 29/08/2024 (root://cmseos.fnal.gov/)
       "T2_US_Vanderbilt",   # 11/08/2024
       "T2_US_Purdue", # 3/07/2024
       "T2_BE_IIHE",  # 3/07/2024
#       "T3_IT_Trieste", # 12/08/2024
    ],
    "2018": [
#        "T3_US_FNALLPC", # 17/07/2024
#        "T1_US_FNAL_Disk", # 17/07/2024
        "T2_US_Vanderbilt", # 27/07/2024
        "T2_US_Purdue",
#        "T2_US_Nebraska",  # 04/08/2024
        "T2_DE_DESY",
        "T2_BE_IIHE", # 27/07/2024
        "T2_CH_CERN",
        "T1_DE_KIT_Disk",
        "T2_DE_RWTH",

#        "T1_FR_CCIN2P3_Tape", # 14/07/2024
        "T2_US_Wisconsin",
#        "T3_IT_Trieste",      # 15/07/2024
#        "T1_FR_CCIN2P3_Disk", # 14/07/2024
        "T2_BE_UCL",
#        "T2_FR_IPHC",
        "T2_PL_Cyfronet",
        "T2_US_Caltech",        
    ],
}


def main():
    with open("das_datasets.json", "r") as f:
        datasets = json.load(f)
        print(datasets)
    for year in ERAS.keys():
        # create a dataset_definition dict for each year
        yreco = f"{year}_UL"
        if not datasets[yreco]:
            continue
        dataset_definition = {}
        for dataset_key, dataset in datasets[yreco].items():
            if isinstance(dataset, list):
                for _dataset, era in zip(dataset, ERAS[year]):
                    dataset_definition[f"/{_dataset}"] = {
                        "short_name": f"{dataset_key}_{era}",
                        "metadata": {"isMC": True},
                    }
            else:
                dataset_definition[f"/{dataset}"] = {
                    "short_name": dataset_key,
                    "metadata": {"isMC": False},
                }
        # the dataset definition is passed to a DataDiscoveryCLI
        ddc = DataDiscoveryCLI()
        # set the allow sites to look for replicas
        ddc.do_allowlist_sites(SITES[year])
        # query rucio and get replicas
        ddc.load_dataset_definition(
            dataset_definition,
            query_results_strategy="all",
            replicas_strategy="round-robin",
        )
        ddc.do_save(f"dataset_discovery_{yreco}.json")
        
        # load and reformat generated fileset
        with open(f"dataset_discovery_{yreco}.json", "r") as f:
            dataset_discovery = json.load(f)
        new_dataset = {key: [] for key in datasets[yreco]}
        for dataset in dataset_discovery:
            root_files = list(dataset_discovery[dataset]["files"].keys())
            dataset_key = dataset_discovery[dataset]["metadata"]["short_name"]
            if dataset_key.startswith("Single") or dataset_key.startswith("MET") or dataset_key.startswith("Tau"):
                new_dataset[dataset_key.split("_")[0]] += root_files
            else:
                new_dataset[dataset_key] = root_files
        # save new fileset and drop 'dataset_discovery' fileset
        os.remove(f"dataset_discovery_{yreco}.json")
        with open(f"fileset_{yreco}_NANO_lxplus.json", "w") as json_file:
            json.dump(new_dataset, json_file, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()