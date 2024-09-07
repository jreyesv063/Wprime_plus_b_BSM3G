import yaml
import importlib.util
from pathlib import Path
from wprime_plus_b.utils.configs.dataset import DatasetConfig


def load_processor_config(config_name: str):
    path = f"wprime_plus_b.configs.processor.{config_name}"
    loader = importlib.util.find_spec(path)

    if loader is None:
        raise Exception(
            f"No config file found for the selected processor '{config_name}'"
        )
    config_module = importlib.import_module(path)
    # this requires that the variables in the config file follow this naming pattern

    config = getattr(
        config_module, "processor_config"
    )  # the module has a variable which is a object with name
    return config


def load_dataset_config(config_name: str):
    configs_path = f"{Path.cwd()}/wprime_plus_b/configs/dataset/datasets_configs.yaml"
    with open(configs_path, "r") as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return DatasetConfig(
        name=configs[config_name],
        nsplit=configs[config_name]["nsplit"],
    )
