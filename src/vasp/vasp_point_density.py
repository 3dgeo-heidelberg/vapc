from vasp import VASP
from data_handler import DATA_HANDLER
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Use VASP.")
    parser.add_argument("laz_file", help="laz_file to mask")
    parser.add_argument("laz_file_out", help="laz_file masked output")
    parser.add_argument("voxel_size", help="voxel_size")
    parser.add_argument("return_at", help="number of additional Voxels searched for in each direction.")
    return parser.parse_args()

if __name__ == "__main__":
    #Usage:
    #python vasp_point_density.py infile outfile voxel_size return_at        
    args = parse_args()
    # laz_file,laz_file_out,mask_file,mask_voxel_size,buffered = args
    dh = DATA_HANDLER(args.laz_file)
    dh.load_las_files()
    
    vasp_pc = VASP(float(args.voxel_size),
                    [0,0,0],
                    return_at = args.return_at)
                    
    vasp_pc.get_data_from_data_handler(dh)
    vasp_pc.compute_point_density()
    vasp_pc.reduce_to_voxels()
    dh.df = vasp_pc.df
    dh.save_as_las(args.laz_file_out)