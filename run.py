import argparse
import os
import random
import sys
import warnings

import numpy as np
import torch

from mfh.data.wild import WildDataset
from mfh.pipeline import MfH
from submodules.metric_depth.zoedepth.utils.arg_utils import parse_unknown

sys.path.append(os.path.join(os.path.dirname(__file__), "submodules/metric_depth"))
warnings.filterwarnings("ignore")


def main(dataset, **kwargs):
    pipeline = MfH(dataset, **kwargs)

    if kwargs.get("input_path") is None:
        pipeline.evaluate()
    else:
        loader = WildDataset(kwargs["input_path"])
        pipeline.infer(loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", type=str, default="wild", choices=["wild", "nyu", "kitti", "ibims", "diode_indoor", "eth3d"]
    )
    parser.add_argument("-s", "--seed", type=int, default=None)
    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    main(dataset=args.dataset, **overwrite_kwargs)
