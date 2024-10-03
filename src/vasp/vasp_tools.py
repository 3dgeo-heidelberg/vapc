from vasp import VASP
from data_handler import DATA_HANDLER

def initiate_vasp(lazfile,
                  voxel_size,
                  origin = [0,0,0]):
    dh = DATA_HANDLER(lazfile)
    dh.load_las_files()
    vasp_pc = VASP(float(voxel_size),
                    origin)      
    vasp_pc.get_data_from_data_handler(dh)
    return vasp_pc, dh

def mask(vasp_pc,
        maskfile,
        segment_in_or_out,
        buffer_size
        ):
    #Apply offset
    vasp_pc.compute_reduction_point()
    dh_mask = DATA_HANDLER(maskfile)
    dh_mask.load_las_files()
    vasp_mask = VASP(vasp_pc.voxel_size,
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
    vasp_mask.compute_voxel_buffer(buffer_size = int(buffer_size))
    #Select by mask
    vasp_pc.select_by_mask(vasp_mask,
                           mask_attribute = "big_int_index",
                           segment_in_or_out = segment_in_or_out)
    #Undo offset
    vasp_pc.compute_offset()
    return vasp_pc.df

def subsample(vasp_pc,
              sub_sample_method = "closest_to_center_of_gravity"):
    vasp_pc.return_at = sub_sample_method
    vasp_pc.reduce_to_voxels()
    return vasp_pc.df


def use_tool(tool_name, infile, outfile, voxel_size, args):
    vasp_pc,dh = initiate_vasp(infile,
                voxel_size,
                origin = [0,0,0])

    if tool_name == "subsample":
        dh.df = subsample(vasp_pc= vasp_pc,
                  sub_sample_method=args["sub_sample_method"]
                  )
    else:
        return "unknown command:%s"%tool_name
    
    dh.save_as_las(outfile)
    
