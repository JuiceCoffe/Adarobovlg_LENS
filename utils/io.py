"""Filesystem, NumPy JSON serialization, and reproducibility helpers."""

import os
import json
import random
import shutil
import numpy as np
import torch


def mkdir_if_not_exists(dir_name: str, recursive: bool = False) -> None:
    """Make directory with the given dir_name
    Args:
        dir_name: input directory name that can be a path
        recursive: recursive directory creation
    """
    if os.path.exists(dir_name):
        return

    if recursive:
        os.makedirs(dir_name)
    else:
        os.mkdir(dir_name)


def rmdir_if_exists(dir_name: str, recursive: bool = False) -> None:
    """Remove directory with the given dir_name
    Args:
        dir_name: input directory name that can be a path
    """
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)


class NumpyArrayEncoder(json.JSONEncoder):
    """
    Python dictionaries can store ndarray array types, but when serialized by dict into JSON files,
    ndarray types cannot be serialized. In order to read and write numpy arrays,
    we need to rewrite JSONEncoder's default method.
    The basic principle is to convert ndarray to list, and then read from list to ndarray.
    """

    def default(self, obj):
        """Serialize NumPy arrays/scalars; delegate unsupported objects to JSONEncoder for TypeError."""
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def dict2json(file_name: str, the_dict: dict) -> None:
    """
    Save dict object as json.
    """
    try:
        json_str = json.dumps(the_dict, cls=NumpyArrayEncoder, indent=4)
        with open(file_name, "w") as json_file:
            json_file.write(json_str)
        return 1
    except (OSError, TypeError, ValueError):
        return 0


def set_seed(seed: int = 42):
    """Seed Python, NumPy, and PyTorch and select deterministic cuDNN behavior."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
