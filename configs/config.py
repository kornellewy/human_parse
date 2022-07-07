from turtle import st
from typing import Union, Dict
from pathlib import Path

import yaml


def read_conf_file(yaml_path: Union[Path, str]) -> dict:
    with open(yaml_path, "r") as stream:
        data_loaded = yaml.safe_load(stream)
    data_loaded = create_tags(data_loaded=data_loaded)
    return data_loaded


def create_tags(data_loaded: dict) -> dict:
    tag = [
        data_loaded["model_generator_name"],
        data_loaded["optim_type"],
    ]
    tag = "_".join(tag)
    data_loaded["tags"] = tag
    return data_loaded


if __name__ == "__main__":
    file_data = read_conf_file("31_acgpn_2/configs/configs/smgm.yaml")
    print(file_data)
