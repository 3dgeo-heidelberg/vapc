import sys, os
from pathlib import Path
current_folder = os.curdir
vasp_path = current_folder+r".\src\vasp"
sys.path.append(vasp_path)  # add vasp directory to PATH
from vasp import VASP
from data_handler import DATA_HANDLER
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Use VASP.")
    parser.add_argument("config_path", help="Path to the configuration JSON file")
    # Add more arguments as needed
    return parser.parse_args()

def main(config_file):
    config_data = open(config_file)
    config = json.load(config_data)

    data_handler = DATA_HANDLER(infiles=config["infiles"],
                                attributes=config["attributes"])
    data_handler.load_las_files()

    vasp = VASP(
        config["voxel_size"],
        config["origin"],
        config["attributes"],
        config["calculate"],
        config["return_at"]
        )
    vasp.get_data_from_data_handler(
        data_handler
        )

    vasp.compute_requested_attributes()
    if "mode" in config["attributes"].values():
        print("!!! WARNING: Using slower version as 'mode' is requested !!!")
        vasp.compute_requested_statistics_per_attributes_numpy()
    else:
        vasp.compute_requested_statistics_per_attributes() 
    vasp.reduce_to_voxels()
    data_handler.df = vasp.df
    data_handler.save_as_las(
        outfile=config["outfile"]
        )
 

if __name__ == "__main__":
    args = parse_args()
    main(args.config_path)
   