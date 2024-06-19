from vasp import VASP
from data_handler import DATA_HANDLER
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Use VASP.")
    parser.add_argument("laz_file", help="file to be subsampled")
    parser.add_argument("laz_file_out", help="output filepath")
    parser.add_argument("voxel_size", help="Voxel size")
    parser.add_argument("sub_sample_method", help="Defines the points the point cloud will be subsampled to")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dh = DATA_HANDLER(args.laz_file)
    dh.load_las_files()
    vasp_pc = VASP(
        float(args.voxel_size),
        [0,0,0],
        return_at = args.sub_sample_method
        )
    vasp_pc.get_data_from_data_handler(dh)
    vasp_pc.reduce_to_voxels()
    dh.df = vasp_pc.df
    dh.save_as_las(args.laz_file_out)