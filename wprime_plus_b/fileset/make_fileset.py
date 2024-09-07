import json
import subprocess

if __name__ == "__main__":
    with open("das_datasets.json", "r") as f:
        datasets = json.load(f)
    reco = "UL"
    filesets = {}
    years = ["2016APV", "2016", "2017", "2018"]
    for year in years:
        yreco = f"{year}_{reco}"
        filesets[yreco] = {}
        for dataset_key, dataset in datasets[yreco].items():
            querys = []
            if isinstance(dataset, list):
                for x in dataset:
                    querys.append(f"file dataset=/{x}")
            else:
                querys.append(f"file dataset=/{dataset}")
            filearray = []
            for newquery in querys:
                farray = subprocess.run(
                    ["dasgoclient", f"-query={newquery}"],
                    stdout=subprocess.PIPE,
                    universal_newlines=True,
                )
                stdout = farray.stdout
                stdout_array = stdout.split("\n")
                stdout_array = stdout_array[:-1]
                stdout_array[-1] = stdout_array[-1].replace(",", "")
                filearray.extend(stdout_array)
            filesets[yreco][dataset_key] = filearray
        with open(f"fileset_{yreco}_NANO.json", "w") as json_file:
            json.dump(filesets[yreco], json_file, indent=4, sort_keys=True)