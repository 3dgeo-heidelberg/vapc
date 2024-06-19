from vasp import VASP
from data_handler import DATA_HANDLER
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Use VASP.")
    parser.add_argument("laz_file", help="file to be subsampled")
    parser.add_argument("laz_file_out", help="output filepath")
    parser.add_argument("voxel_size", help="Voxel size")
    parser.add_argument("filter_attribute", help="Defines the attribute to apply the filter on.")
    parser.add_argument("min_max_eq", help="Choose if value is upper or lowerthreshold. Or if it has to be equal to.")
    parser.add_argument("filter_value", help="If given, will remove voxels represented by less than the given number of points")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dh = DATA_HANDLER(args.laz_file)
    dh.load_las_files()
    vasp_pc = VASP(
        float(args.voxel_size),
        [0,0,0],
        compute = [args.filter_attribute]
        )
    vasp_pc.get_data_from_data_handler(dh)
    vasp_pc.compute_point_count()
    # vasp_pc.reduce_to_voxels()
    vasp_pc.filter_attributes(
                        args.filter_attribute,
                        args.min_max_eq,
                        int(args.filter_value),
                        )
    dh.df = vasp_pc.df.drop([args.filter_attribute,"voxel_x","voxel_y","voxel_z"], axis=1)
    dh.save_as_las(args.laz_file_out)