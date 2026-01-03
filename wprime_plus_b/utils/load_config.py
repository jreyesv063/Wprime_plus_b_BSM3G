import yaml
import importlib.util
from pathlib import Path

def load_processor_config(processor: str, channel: str, lepton_flavor: str):

    class ProcessorConfig:
        def __init__(self, name, channel, lepton_flavor):
            self.name = name
            self.channel = channel
            self.lepton_flavor = lepton_flavor

    return ProcessorConfig(
        name=processor,
        channel=channel,
        lepton_flavor=lepton_flavor,
    )

def load_dataset_config(config_name: str, object_syst: str):

    class DatasetConfig:
        def __init__(self, name, nsplit):
            self.name = name
            self.nsplit = nsplit

    object_systematic_variation = (object_syst.lower() == "true")
    configs_file= "datasets_configs_systematics.yaml" if object_systematic_variation else "datasets_configs.yaml"
    configs_path = f"{Path.cwd()}/wprime_plus_b/configs/dataset/{configs_file}"

    with open(configs_path, "r") as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return DatasetConfig(
        name=configs[config_name],
        nsplit=configs[config_name]["nsplit"],
    )
