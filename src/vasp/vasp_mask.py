from vasp import VASP
from data_handler import DATA_HANDLER
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Use VASP.")
    parser.add_argument("laz_file", help="laz_file to mask")
    parser.add_argument("laz_file_out", help="laz_file masked output")
    parser.add_argument("mask_file", help="lazmask")
    parser.add_argument("mask_voxel_size", help="voxel_size")
    parser.add_argument("buffer_size", help="number of additional Voxels searched for in each direction.")
    return parser.parse_args()

if __name__ == "__main__":
    #Usage:
    #python vasp_mask.py infile outfile maskfile voxel_size buffer_size        
    args = parse_args()
    # laz_file,laz_file_out,mask_file,mask_voxel_size,buffered = args
    dh = DATA_HANDLER(args.laz_file)
    dh.load_las_files()
    
    vasp_pc = VASP(float(args.mask_voxel_size),
                    [0,0,0])
                    
    vasp_pc.get_data_from_data_handler(dh)
    #Apply offset
    vasp_pc.compute_reduction_point()
    
    dh_mask = DATA_HANDLER([args.mask_file])
                            
    dh_mask.load_las_files()
    vasp_mask = VASP(float(args.mask_voxel_size),
                    [0,0,0])
    vasp_mask.get_data_from_data_handler(dh_mask)
    #Apply offset
    vasp_mask.compute_reduction_point()
    min_reduction_point = (min([vasp_pc.reduction_point[0],vasp_mask.reduction_point[0]]),min([vasp_pc.reduction_point[1],vasp_mask.reduction_point[1]]),min([vasp_pc.reduction_point[2],vasp_mask.reduction_point[2]]))
    # print(min_reduction_point)
    vasp_pc.reduction_point = min_reduction_point
    vasp_pc.compute_offset()
    vasp_mask.reduction_point = min_reduction_point
    vasp_mask.compute_offset()

    #Buffer mask voxelized point cloud
    vasp_mask.compute_voxel_buffer(buffer_size = int(args.buffer_size))
    
    #Select by mask
    vasp_pc.select_by_mask(vasp_mask,"big_int_index")
    #Undo offset
    vasp_pc.compute_offset()
    #Save Point Cloud
    dh.df = vasp_pc.df
    dh.save_as_las(args.laz_file_out)