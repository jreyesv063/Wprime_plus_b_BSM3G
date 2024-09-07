import re
import json


def main():
    """
    builds a dictionary that maps each sample name to a simplified form
    """
    simplified_samples = {}
    for year in ["2016", "2017", "2018"]:
        fileset_path = f"/home/cms-jovyan/wprime_plus_b/wprime_plus_b/fileset/fileset_{year}_UL_NANO.json"
        with open(fileset_path, "r") as f:
            fileset = json.load(f)
        simplified_samples[year] = {sample: "" for sample in fileset}

    pattern = "(.*)_Match|(.*)_Tune"
    for year, samples in simplified_samples.items():
        for sample in samples:
            match = re.match(pattern, sample)
            if match:
                sample_match = match[2]
                if not sample_match:
                    sample_match = match[1]
            else:
                sample_match = sample

            simplified_samples[year][sample] = sample_match

    output_path = (
        "/home/cms-jovyan/wprime_plus_b/wprime_plus_b/data/simplified_samples.json"
    )
    with open(output_path, "w") as f:
        json.dump(simplified_samples, f)


if __name__ == "__main__":
    main()
