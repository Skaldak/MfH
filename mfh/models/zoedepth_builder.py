from importlib import import_module


def build_model(config):
    module_name = f"mfh.models.{config.model}"
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as e:
        # print the original error message
        print(e)
        raise ValueError(f"Model {config.model} not found. Refer above error for details.") from e
    try:
        get_version = getattr(module, "get_version")
    except AttributeError as e:
        raise ValueError(f"Model {config.model} has no get_version function.") from e
    return get_version(config.version_name).build_from_config(config)
