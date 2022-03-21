import json
import os
import glob
import copy

cfg_dict = {}
cfg_dirnames = []
cfg_dirnames.append(os.path.join("eagent", "configs", "ewalker"))
cfg_dirnames.append(os.path.join("eagent", "configs", "ehand_egg"))
default_filename = "default.json"

for cfg_dirname in cfg_dirnames:
    with open(os.path.join(cfg_dirname, default_filename), "r") as f:
        default_cfg = json.load(f)

    for filename in glob.glob(os.path.join(cfg_dirname, "*")):
        if os.path.basename(filename) == default_filename:
            continue
        with open(filename, "r") as f:
            cfg_dict[os.path.basename(filename)] = copy.deepcopy(default_cfg)
            cfg_dict[os.path.basename(filename)].update(json.load(f))


if __name__ == "__main__":
    for k, v in cfg_dict.items():
        print(k)
