import json
import os
import platform

from submodules.metric_depth.zoedepth.utils.arg_utils import infer_type
from submodules.metric_depth.zoedepth.utils.easydict import EasyDict as edict

DATA_ROOT = "data"

COMMON_CONFIG = {
    "dataset": "wild",
    "encoder": "vitl",
    "debug": False,
    "shuffle": False,
    "upsample": False,
    "visualize": False,
    "input_path": "data/wild",
    "output_path": "logs/wild",
    "version_name": "relative",
}

DATASETS_CONFIG = {
    "wild": {
        "dataset": "wild",
        "eigen_crop": False,
        "garg_crop": False,
        "do_kb_crop": False,
        "min_depth_eval": 0,
        "max_depth_eval": 10,
        "min_depth": 1e-3,
        "max_depth": 10,
    },
    "nyu": {
        "dataset": "nyu",
        "min_depth": 1e-3,
        "max_depth": 10,
        "data_path_eval": os.path.join(DATA_ROOT, "nyu_depth_v2/official_splits/test"),
        "gt_path_eval": os.path.join(DATA_ROOT, "nyu_depth_v2/official_splits/test"),
        "filenames_file_eval": "submodules/metric_depth/train_test_inputs/nyudepthv2_test_files_with_gt.txt",
        "min_depth_eval": 1e-3,
        "max_depth_eval": 10,
        "do_kb_crop": False,
        "garg_crop": False,
        "eigen_crop": True,
    },
    "kitti": {
        "dataset": "kitti",
        "min_depth": 0.001,
        "max_depth": 80,
        "data_path_eval": os.path.join(DATA_ROOT, "kitti/raw"),
        "gt_path_eval": os.path.join(DATA_ROOT, "kitti/gts"),
        "filenames_file_eval": "submodules/metric_depth/train_test_inputs/kitti_eigen_test_files_with_gt.txt",
        "min_depth_eval": 1e-3,
        "max_depth_eval": 80,
        "do_kb_crop": True,
        "garg_crop": True,
        "eigen_crop": False,
        "scale": False,
        "offset": True,
    },
    "ibims": {
        "dataset": "ibims",
        "ibims_root": os.path.join(DATA_ROOT, "ibims1_core_raw"),
        "eigen_crop": True,
        "garg_crop": False,
        "do_kb_crop": False,
        "min_depth_eval": 0,
        "max_depth_eval": 10,
        "min_depth": 1e-3,
        "max_depth": 10,
    },
    "diode_indoor": {
        "dataset": "diode_indoor",
        "diode_indoor_root": os.path.join(DATA_ROOT, "diode/val/indoors"),
        "eigen_crop": True,
        "garg_crop": False,
        "do_kb_crop": False,
        "min_depth_eval": 1e-3,
        "max_depth_eval": 10,
        "min_depth": 1e-3,
        "max_depth": 10,
    },
    "eth3d": {
        "dataset": "eth3d",
        "min_depth": 1e-3,
        "max_depth": 80,
        "eth3d_root": os.path.join(DATA_ROOT, "eth3d"),
        "eigen_crop": False,
        "garg_crop": False,
        "do_kb_crop": False,
        "min_depth_eval": 1e-3,
        "max_depth_eval": 80,
    },
}


def flatten(config, except_keys=("bin_conf")):
    def recurse(inp):
        if isinstance(inp, dict):
            for key, value in inp.items():
                if key in except_keys:
                    yield (key, value)
                if isinstance(value, dict):
                    yield from recurse(value)
                else:
                    yield (key, value)

    return dict(list(recurse(config)))


def split_combined_args(kwargs):
    """Splits the arguments that are combined with '__' into multiple arguments.
       Combined arguments should have equal number of keys and values.
       Keys are separated by '__' and Values are separated with ';'.
       For example, '__n_bins__lr=256;0.001'

    Args:
        kwargs (dict): key-value pairs of arguments where key-value is optionally combined according to the above format.

    Returns:
        dict: Parsed dict with the combined arguments split into individual key-value pairs.
    """
    new_kwargs = dict(kwargs)
    for key, value in kwargs.items():
        if key.startswith("__"):
            keys = key.split("__")[1:]
            values = value.split(";")
            assert len(keys) == len(
                values
            ), f"Combined arguments should have equal number of keys and values. Keys are separated by '__' and Values are separated with ';'. For example, '__n_bins__lr=256;0.001. Given (keys,values) is ({keys}, {values})"
            for k, v in zip(keys, values):
                new_kwargs[k] = v
    return new_kwargs


def parse_list(config, key, dtype=int):
    """Parse a list of values for the key if the value is a string. The values are separated by a comma.
    Modifies the config in place.
    """
    if key in config:
        if isinstance(config[key], str):
            config[key] = list(map(dtype, config[key].split(",")))
        assert isinstance(config[key], list) and all(
            [isinstance(e, dtype) for e in config[key]]
        ), f"{key} should be a list of values dtype {dtype}. Given {config[key]} of type {type(config[key])} with values of type {[type(e) for e in config[key]]}."


def get_model_config(model_name, model_version=None):
    """Find and parse the .json config file for the model.

    Args:
        model_name (str): name of the model. The config file should be named config_{model_name}[_{model_version}].json under the models/{model_name} directory.
        model_version (str, optional): Specific config version. If specified config_{model_name}_{model_version}.json is searched for and used. Otherwise config_{model_name}.json is used. Defaults to None.

    Returns:
        easydict: the config dictionary for the model.
    """
    config_fname = (
        f"config_{model_name}_{model_version}.json" if model_version is not None else f"config_{model_name}.json"
    )
    config_file = os.path.join("mfh", "models", model_name, config_fname)
    if not os.path.exists(config_file):
        return None

    with open(config_file, "r") as f:
        config = edict(json.load(f))

    # handle dictionary inheritance
    # only training config is supported for inheritance
    if "inherit" in config.train and config.train.inherit is not None:
        inherit_config = get_model_config(config.train["inherit"]).train
        for key, value in inherit_config.items():
            if key not in config.train:
                config.train[key] = value
    return edict(config)


def update_model_config(config, mode, model_name, model_version=None, strict=False):
    model_config = get_model_config(model_name, model_version)
    if model_config is not None:
        config = {**config, **flatten({**model_config.model, **model_config[mode]})}
    elif strict:
        raise ValueError(f"Config file for model {model_name} not found.")
    return config


def check_choices(name, value, choices):
    if value not in choices:
        raise ValueError(f"{name} {value} not in supported choices {choices}")


def get_mfh_config():
    mfh_config = dict(
        paint_batch_size=8,
        paint_total_size=32,
        total_iter=50,
        loss_fn="silog",
        scale=True,
        offset=True,
        inverse=False,
        inverse_gt=True,
        min_ratio=0.2,
        max_ratio=0.8,
        align=True,
        visualize=False,
    )

    return edict(mfh_config)


def get_config(model_name, mode="train", dataset=None, **overwrite_kwargs):
    """Main entry point to get the config for the model.

    Args:
        model_name (str): name of the desired model.
        mode (str, optional): "train" or "infer". Defaults to 'train'.
        dataset (str, optional): If specified, the corresponding dataset configuration is loaded as well. Defaults to None.

    Keyword Args: key-value pairs of arguments to overwrite the default config.

    The order of precedence for overwriting the config is (Higher precedence first):
        # 1. overwrite_kwargs
        # 2. "config_version": Config file version if specified in overwrite_kwargs. The corresponding config loaded is config_{model_name}_{config_version}.json
        # 3. "version_name": Default Model version specific config specified in overwrite_kwargs. The corresponding config loaded is config_{model_name}_{version_name}.json
        # 4. common_config: Default config for all models specified in COMMON_CONFIG

    Returns:
        easydict: The config dictionary for the model.
    """

    check_choices("Model", model_name, ["zoedepth", "zoedepth_nk"])
    check_choices("Mode", mode, ["train", "infer", "eval"])
    if mode == "train":
        check_choices("Dataset", dataset, ["nyu", "kitti", "mix", None])

    config = flatten({**COMMON_CONFIG})
    config = update_model_config(config, mode, model_name, model_version=config["version_name"])

    # update with model version specific config
    version_name = overwrite_kwargs.get("version_name", config["version_name"])
    config = update_model_config(config, mode, model_name, model_version=version_name)

    # update with config version if specified
    config_version = overwrite_kwargs.get("config_version", None)
    if config_version is not None:
        print("Overwriting config with config_version", config_version)
        config = update_model_config(config, mode, model_name, config_version)

    config = {**config, **get_mfh_config()}

    if dataset is not None:
        config["dataset"] = dataset
        config = {**config, **DATASETS_CONFIG[dataset]}

    # update with overwrite_kwargs
    # Combined args are useful for hyperparameter search
    overwrite_kwargs = split_combined_args(overwrite_kwargs)
    config = {**config, **overwrite_kwargs}

    # Model specific post processing of config
    parse_list(config, "n_attractors")

    config["model"] = model_name
    typed_config = {k: infer_type(v) for k, v in config.items()}
    # add hostname to config
    config["hostname"] = platform.node()
    return edict(typed_config)
