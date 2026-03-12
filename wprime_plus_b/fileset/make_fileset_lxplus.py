import os
import json
import argparse
from coffea.dataset_tools.dataset_query import DataDiscoveryCLI

# Important: Sample information can be found in https://opendata.cern.ch/

ERAS = {
    "2016APV": ["B1", "B2", "C", "D", "E","F"],
    "2016": ["F", "G", "H"],
    "2017": ["B", "C", "D", "E", "F"],
    "2018": ["A", "B", "C", "D"],
    "2022_pre": ["C", "D"],
    "2022_post": ["E", "F", "G"],
    "2023_pre": ["C"],
    "2023_post": ["D"],
    "2024": ["C", "D", "E", "F", "G", "H", "I"],
}

SITES = {
    "2016APV": [
        # ---- T1 sites ----
        "T1_DE_KIT_Disk",
        "T1_ES_PIC_Disk",
        "T1_FR_CCIN2P3_Disk",
        "T1_FR_CCIN2P3_Tape",
    #   "T1_RU_JINR_Disk", # 24/02/2026
        "T1_UK_RAL_Disk",
        "T1_US_FNAL_Disk",
        # ---- T2 sites ----
    #   "T2_BE_IIHE", # 24/02/2026
        "T2_BR_UERJ",
        "T2_CH_CERN",
        "T2_DE_DESY",
    #   "T2_DE_RWTH", # 24/02/2026
        "T2_FR_IPHC",
    #   "T2_IN_TIFR",
        "T2_IT_Legnaro",
        "T2_IT_Rome",
    #   "T2_TW_NCHC",
    #   "T2_UK_London_IC", # 24/02/2026
        "T2_US_MIT",
        "T2_US_Nebraska",
        "T2_US_Purdue",
    #   "T2_US_Vanderbilt",
        "T3_CH_CERN_OpenData",
        # ---- T3 sites ----
        "T3_FR_IPNL",
        "T3_IT_Trieste",
    #   "T3_KR_UOS",
    #   "T3_US_FNALLPC"
    ],
    "2016": [
        # ---- T1 sites ----
    #   "T1_DE_KIT_Disk",
    #   "T1_ES_PIC_Disk",
        "T1_FR_CCIN2P3_Disk",
        "T1_FR_CCIN2P3_Tape",
    #    "T1_RU_JINR_Disk",
    #   "T1_UK_RAL_Disk",
        "T1_US_FNAL_Disk",
        # ---- T2 sites ----
    #    "T2_BE_IIHE",
    #    "T2_BR_UERJ",
        "T2_CH_CERN",
        "T2_DE_DESY",
        "T2_DE_RWTH",
        "T2_FR_IPHC",
    #   "T2_IN_TIFR",
        "T2_IT_Legnaro",
        "T2_IT_Rome",
    #   "T2_TW_NCHC",
    #    "T2_UK_London_IC", #24/02/2026
        "T2_US_MIT",
    #   "T2_US_Nebraska",
    #   "T2_US_Purdue",
    #   "T2_US_Vanderbilt",
        "T3_CH_CERN_OpenData",
        # ---- T3 sites ----
        "T3_FR_IPNL",
        "T3_IT_Trieste",
    #   "T3_KR_UOS",
        "T3_US_FNALLPC"
    ],
    "2017": [
        # ---- T1 sites ----
        "T1_FR_CCIN2P3_Tape",
    #    "T1_RU_JINR_Disk", # Hoy
        "T1_US_FNAL_Disk", 
        #---- T2 sites ----
        #"T2_BE_IIHE",
        "T2_BE_UCL",
        "T2_CH_CERN",
    #    "T2_DE_DESY",  # Hoy
        "T2_DE_RWTH",
        #"T2_EE_Estonia",
    #    "T2_ES_CIEMAT", # 28/02/2026
        "T2_FR_IPHC",
        "T2_HU_Budapest",
    #    "T2_UK_London_IC", # 22/02/2026
        "T2_US_MIT",       
    #    "T2_US_Nebraska",
        "T2_US_Purdue",
        "T2_US_Vanderbilt",
        # ---- T3 sites ----
    #    "T3_CH_PSI",
        "T3_FR_IPNL",
        "T3_IT_Trieste",
    #    "T3_KR_KISTI",  # Este por Purpue
        #"T3_KR_UOS",
        #"T3_US_Baylor",
        "T3_US_FNALLPC",
        "T3_US_NotreDame"
    ],
    "2018": [
        # ---- T1 sites ----
        "T1_DE_KIT_Disk",
        "T1_FR_CCIN2P3_Disk",
        #"T1_FR_CCIN2P3_Tape", 
        #"T1_IT_CNAF_Disk",
        #"T1_RU_JINR_Disk", # 24/02/2026
        "T1_UK_RAL_Disk",
        "T1_US_FNAL_Disk",
        #---- T2 sites ----
        #"T2_BE_IIHE",
        "T2_BE_UCL",
        #"T2_BR_SPRACE", # 22/02/2026
        "T2_CH_CERN",
        "T2_DE_DESY",
        "T2_DE_RWTH",
        "T2_FR_IPHC",
        "T2_HU_Budapest",
        "T2_IT_Rome",
        "T2_PL_Cyfronet",
        "T2_UK_London_IC",
        #"T2_US_Caltech",
        #"T2_US_Nebraska",
        #"T2_US_Purdue",
        #"T2_US_Vanderbilt",
        "T2_US_Wisconsin",
        # ---- T3 sites ----
        "T3_IT_Trieste",
        "T3_US_FNALLPC"
    ],
    "2022_pre": [
        # ---- T1 sites ----
        "T1_DE_KIT_Disk",
        "T1_DE_KIT_Tape",
        "T1_ES_PIC_Disk",
        "T1_FR_CCIN2P3_Disk",
        "T1_FR_CCIN2P3_Tape",
        "T1_IT_CNAF_Disk",
        "T1_IT_CNAF_Tape",
        "T1_PL_NCBJ_Disk",
        "T1_RU_JINR_Disk",
        "T1_UK_RAL_Disk",
        "T1_UK_RAL_Tape",
        "T1_US_FNAL_Disk",
        "T1_US_FNAL_Tape",
        # ---- T2 sites ----
        "T2_BE_IIHE",
        "T2_BE_UCL",
        "T2_BR_SPRACE",
        "T2_CH_CERN",
        "T2_CH_CSCS",
        "T2_CN_Beijing",
        "T2_DE_DESY",
        "T2_DE_RWTH",
        "T2_EE_Estonia",
        "T2_ES_CIEMAT",
        "T2_FI_HIP",
        "T2_FR_GRIF",
        "T2_FR_IPHC",
        "T2_HU_Budapest",
        "T2_IN_TIFR",
        "T2_IT_Bari",
        "T2_IT_Legnaro",
        "T2_IT_Pisa",
        "T2_IT_Rome",
        "T2_PL_Cyfronet",
        "T2_TR_METU",
        "T2_UA_KIPT",
        "T2_UK_London_Brunel",
        "T2_UK_London_IC",
        "T2_UK_SGrid_RALPP",
        "T2_US_Caltech",
        "T2_US_Florida",
        "T2_US_Nebraska",
        "T2_US_Purdue",
        "T2_US_UCSD",
        "T2_US_Vanderbilt",
        "T2_US_Wisconsin",
        # ---- T3 sites ----        
        "T3_FR_IPNL",
        "T3_KR_KISTI",
        "T3_KR_UOS",
        "T3_US_NotreDame",
        "T3_US_Rutgers"
    ],
    "2022_post": [
         # ---- T1 sites ----
        "T1_DE_KIT_Disk",
        "T1_DE_KIT_Tape",
        "T1_ES_PIC_Disk",
        "T1_FR_CCIN2P3_Disk",
        "T1_FR_CCIN2P3_Tape",
        "T1_IT_CNAF_Disk",
        "T1_IT_CNAF_Tape",
        "T1_PL_NCBJ_Disk",
        "T1_RU_JINR_Disk",
        "T1_UK_RAL_Disk",
        "T1_UK_RAL_Tape",
        "T1_US_FNAL_Disk",
        "T1_US_FNAL_Tape",
        # ---- T2 sites ----
        "T2_BE_IIHE",
        "T2_BE_UCL",
        "T2_BR_SPRACE",
        "T2_CH_CERN",
        "T2_CH_CSCS",
        "T2_CN_Beijing",
        "T2_DE_DESY",
        "T2_DE_RWTH",
        "T2_EE_Estonia",
        "T2_ES_CIEMAT",
        "T2_FI_HIP",
        "T2_FR_GRIF",
        "T2_FR_IPHC",
        "T2_HU_Budapest",
        "T2_IN_TIFR",
        "T2_IT_Bari",
        "T2_IT_Legnaro",
        "T2_IT_Pisa",
        "T2_IT_Rome",
        "T2_PL_Cyfronet",
        "T2_TR_METU",
        "T2_UA_KIPT",
        "T2_UK_London_Brunel",
        "T2_UK_London_IC",
        "T2_UK_SGrid_RALPP",
        "T2_US_Caltech",
        "T2_US_Florida",
        "T2_US_Nebraska",
        "T2_US_Purdue",
        "T2_US_UCSD",
        "T2_US_Vanderbilt",
        "T2_US_Wisconsin",
        # ---- T3 sites ----        
        "T3_FR_IPNL",
        "T3_KR_KISTI",
        "T3_KR_UOS",
        "T3_US_NotreDame",
        "T3_US_Rutgers"
    ],
    "2023_pre": [
         # ---- T1 sites ----
        "T1_DE_KIT_Disk",
        "T1_DE_KIT_Tape",
        "T1_ES_PIC_Disk",
        "T1_FR_CCIN2P3_Disk",
        "T1_FR_CCIN2P3_Tape",
        "T1_IT_CNAF_Disk",
        "T1_IT_CNAF_Tape",
        "T1_PL_NCBJ_Disk",
        "T1_RU_JINR_Disk",
        "T1_UK_RAL_Disk",
        "T1_UK_RAL_Tape",
        "T1_US_FNAL_Disk",
        "T1_US_FNAL_Tape",
        # ---- T2 sites ----
        "T2_BE_IIHE",
        "T2_BE_UCL",
        "T2_BR_SPRACE",
        "T2_CH_CERN",
        "T2_CH_CSCS",
        "T2_CN_Beijing",
        "T2_DE_DESY",
        "T2_DE_RWTH",
        "T2_EE_Estonia",
        "T2_ES_CIEMAT",
        "T2_FI_HIP",
        "T2_FR_GRIF",
        "T2_FR_IPHC",
        "T2_HU_Budapest",
        "T2_IN_TIFR",
        "T2_IT_Bari",
        "T2_IT_Legnaro",
        "T2_IT_Pisa",
        "T2_IT_Rome",
        "T2_PL_Cyfronet",
        "T2_TR_METU",
        "T2_UA_KIPT",
        "T2_UK_London_Brunel",
        "T2_UK_London_IC",
        "T2_UK_SGrid_RALPP",
        "T2_US_Caltech",
        "T2_US_Florida",
        "T2_US_Nebraska",
        "T2_US_Purdue",
        "T2_US_UCSD",
        "T2_US_Vanderbilt",
        "T2_US_Wisconsin",
        # ---- T3 sites ----        
        "T3_FR_IPNL",
        "T3_KR_KISTI",
        "T3_KR_UOS",
        "T3_US_NotreDame",
        "T3_US_Rutgers"
    ],
    "2023_post": [
         # ---- T1 sites ----
        "T1_DE_KIT_Disk",
        "T1_DE_KIT_Tape",
        "T1_ES_PIC_Disk",
        "T1_FR_CCIN2P3_Disk",
        "T1_FR_CCIN2P3_Tape",
        "T1_IT_CNAF_Disk",
        "T1_IT_CNAF_Tape",
        "T1_PL_NCBJ_Disk",
        "T1_RU_JINR_Disk",
        "T1_UK_RAL_Disk",
        "T1_UK_RAL_Tape",
        "T1_US_FNAL_Disk",
        "T1_US_FNAL_Tape",
        # ---- T2 sites ----
        "T2_BE_IIHE",
        "T2_BE_UCL",
        "T2_BR_SPRACE",
        "T2_CH_CERN",
        "T2_CH_CSCS",
        "T2_CN_Beijing",
        "T2_DE_DESY",
        "T2_DE_RWTH",
        "T2_EE_Estonia",
        "T2_ES_CIEMAT",
        "T2_FI_HIP",
        "T2_FR_GRIF",
        "T2_FR_IPHC",
        "T2_HU_Budapest",
        "T2_IN_TIFR",
        "T2_IT_Bari",
        "T2_IT_Legnaro",
        "T2_IT_Pisa",
        "T2_IT_Rome",
        "T2_PL_Cyfronet",
        "T2_TR_METU",
        "T2_UA_KIPT",
        "T2_UK_London_Brunel",
        "T2_UK_London_IC",
        "T2_UK_SGrid_RALPP",
        "T2_US_Caltech",
        "T2_US_Florida",
        "T2_US_Nebraska",
        "T2_US_Purdue",
        "T2_US_UCSD",
        "T2_US_Vanderbilt",
        "T2_US_Wisconsin",
        # ---- T3 sites ----        
        "T3_FR_IPNL",
        "T3_KR_KISTI",
        "T3_KR_UOS",
        "T3_US_NotreDame",
        "T3_US_Rutgers"
    ],
    "2024": [
         # ---- T1 sites ----
        "T1_DE_KIT_Disk",
        "T1_DE_KIT_Tape",
        "T1_ES_PIC_Disk",
        "T1_FR_CCIN2P3_Disk",
        "T1_FR_CCIN2P3_Tape",
        "T1_IT_CNAF_Disk",
        "T1_IT_CNAF_Tape",
        "T1_PL_NCBJ_Disk",
        "T1_RU_JINR_Disk",
        "T1_UK_RAL_Disk",
        "T1_UK_RAL_Tape",
        "T1_US_FNAL_Disk",
        "T1_US_FNAL_Tape",
        # ---- T2 sites ----
        "T2_BE_IIHE",
        "T2_BE_UCL",
        "T2_BR_SPRACE",
        "T2_CH_CERN",
        "T2_CH_CSCS",
        "T2_CN_Beijing",
        "T2_DE_DESY",
        "T2_DE_RWTH",
        "T2_EE_Estonia",
        "T2_ES_CIEMAT",
        "T2_FI_HIP",
        "T2_FR_GRIF",
        "T2_FR_IPHC",
        "T2_HU_Budapest",
        "T2_IN_TIFR",
        "T2_IT_Bari",
        "T2_IT_Legnaro",
        "T2_IT_Pisa",
        "T2_IT_Rome",
        "T2_PL_Cyfronet",
        "T2_TR_METU",
        "T2_UA_KIPT",
        "T2_UK_London_Brunel",
        "T2_UK_London_IC",
        "T2_UK_SGrid_RALPP",
        "T2_US_Caltech",
        "T2_US_Florida",
        "T2_US_Nebraska",
        "T2_US_Purdue",
        "T2_US_UCSD",
        "T2_US_Vanderbilt",
        "T2_US_Wisconsin",
        # ---- T3 sites ----        
        "T3_FR_IPNL",
        "T3_KR_KISTI",
        "T3_KR_UOS",
        "T3_US_NotreDame",
        "T3_US_Rutgers"
    ]
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate coffea datasets per year"
    )
    parser.add_argument(
        "--year",
        choices=ERAS.keys(),
        help="Process only a specific year (e.g. 2016APV, 2016, 2017, 2018, 2022_pre, 2022_post, 2023_pre, 2023_post, 2024)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open("das_datasets.json", "r") as f:
        datasets = json.load(f)

    years_to_run = [args.year] if args.year else ERAS.keys()

    for year in years_to_run:
        if year in ["2016APV", "2016", "2017", "2018"]:
            yreco = f"{year}_UL"
        else:
            yreco = year

        if yreco not in datasets or not datasets[yreco]:
            print(f"[SKIP] No datasets for {yreco}")
            continue

        print(f"[INFO] Processing {yreco}")

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

        ddc = DataDiscoveryCLI()
        ddc.do_allowlist_sites(SITES[year])
        ddc.load_dataset_definition(
            dataset_definition,
            query_results_strategy="all",
            replicas_strategy="round-robin",
        )
        ddc.do_save(f"dataset_discovery_{yreco}.json")

        with open(f"dataset_discovery_{yreco}.json", "r") as f:
            dataset_discovery = json.load(f)

        new_dataset = {key: [] for key in datasets[yreco]}
        for dataset in dataset_discovery:
            root_files = list(dataset_discovery[dataset]["files"].keys())
            dataset_key = dataset_discovery[dataset]["metadata"]["short_name"]
            if dataset_key.startswith(("Single", "MET", "Tau")):
                new_dataset[dataset_key.split("_")[0]] += root_files
            else:
                new_dataset[dataset_key] = root_files

        os.remove(f"dataset_discovery_{yreco}.json")
        with open(f"fileset_{yreco}_NANO_lxplus.json", "w") as json_file:
            json.dump(new_dataset, json_file, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
