from .vasp import VASP
from .data_handler import DATA_HANDLER

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
        buffer_size,
        reduce_to = False
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
                           mask_attribute = "voxel_index",
                           segment_in_or_out = segment_in_or_out)
    #Undo offset
    vasp_pc.compute_offset()
    if reduce_to: #check if it should be reduced to voxels
        vasp_pc.return_at = reduce_to
        vasp_pc.reduce_to_voxels()
    return vasp_pc.df

def subsample(vasp_pc,
              reduce_to = "closest_to_center_of_gravity"):
    vasp_pc.return_at = reduce_to
    vasp_pc.reduce_to_voxels()
    return vasp_pc.df

def compute_attributes(vasp_pc,
                       compute,
                       reduce_to):
    vasp_pc.compute = compute
    vasp_pc.compute_requested_attributes()
    if reduce_to: #check if it should be reduced to voxels
        vasp_pc.return_at = reduce_to
        vasp_pc.reduce_to_voxels()
    return vasp_pc.df

def filter_by_attributes(
        vasp_pc,
        filters,
        reduce_to):

    #Compute filter attributes
    vasp_pc.compute = list(filters.keys())
    vasp_pc.compute_requested_attributes()
    #Filter attribute
    for fa in filters.keys():
        filter_attribute = filters[fa]
        for filter_condition in filter_attribute.keys():
            if filter_condition == "equal":
                vasp_pc.filter_attributes(
                            fa,
                            "eq",
                            filter_attribute[filter_condition],
                            )
            elif filter_condition == "bigger_equal":
                vasp_pc.filter_attributes(
                            fa,
                            "min_eq",
                            filter_attribute[filter_condition],
                            )
            elif filter_condition == "bigger":
                vasp_pc.filter_attributes(
                            fa,
                            "min",
                            filter_attribute[filter_condition],
                            )
            elif filter_condition == "smaller_equal":
                vasp_pc.filter_attributes(
                            fa,
                            "max_eq",
                            filter_attribute[filter_condition],
                            )
            elif filter_condition == "smaller":
                vasp_pc.filter_attributes(
                            fa,
                            "max",
                            filter_attribute[filter_condition],
                            )
            else:
                return False
    if reduce_to: #check if it should be reduced to voxels
        vasp_pc.return_at = reduce_to
        vasp_pc.reduce_to_voxels()
    return vasp_pc.df


def filter_by_attributes_and_compute(
        vasp_pc,
        filters,
        compute,
        reduce_to):
    #Lets filter first
    vasp_pc.df = filter_by_attributes(
        vasp_pc,
        filters,
        reduce_to = False)
    
    #Prevent multiple computations of the same attribute:
    for filter_attr in filters.keys():
        compute = list(filter((filter_attr).__ne__, compute))
    
    #And compute attributes after
    return compute_attributes(vasp_pc,
                       compute,
                       reduce_to)

def compute_statistics(
        vasp_pc,
        statistics,
        reduce_to):
    vasp_pc.attributes = statistics
    vasp_pc.compute_requested_statistics_per_attributes()
    if reduce_to: #check if it should be reduced to voxels
        vasp_pc.return_at = reduce_to
        vasp_pc.reduce_to_voxels()
    return vasp_pc.df
        



def use_tool(tool_name, 
             infile, 
             outfile, 
             voxel_size, 
             args,
             reduce_to):
    
    vasp_pc,dh = initiate_vasp(infile,
                voxel_size,
                origin = [0,0,0])

    if tool_name == "subsample":
        dh.df = subsample(vasp_pc= vasp_pc,
                  reduce_to=args["sub_sample_method"]
                  )
        
    elif tool_name == "mask":
        dh.df = mask(vasp_pc=vasp_pc,
                     maskfile=args["maskfile"],
                     segment_in_or_out=args["segment_in_or_out"],
                     buffer_size=args["buffer_size"],
                     reduce_to=reduce_to)
        
    elif tool_name == "compute":
        dh.df = compute_attributes(vasp_pc = vasp_pc,
                                   compute = args["compute"],
                                   reduce_to = reduce_to)
        
    elif tool_name == "filter":
        dh.df = filter_by_attributes(vasp_pc = vasp_pc,
                                   filters = args["filters"],
                                   reduce_to = reduce_to)
        
    elif tool_name == "filter_and_compute":
        dh.df = filter_by_attributes_and_compute(vasp_pc = vasp_pc,
                                   filters = args["filters"],
                                   compute = args["compute"],
                                   reduce_to = reduce_to)
    elif tool_name == "statistics":
        dh.df = compute_statistics(vasp_pc = vasp_pc,
                                   statistics = args["statistics"],
                                   reduce_to = reduce_to)
    else:
        return "unknown command:%s"%tool_name
    dh.save_as_las(outfile)
    
def laSZ_to_ply(infile,
                outfile,
                voxel_size,
                shift_to_center = False):
    dh = DATA_HANDLER(infile)
    dh.load_las_files()
    dh.save_as_ply(outfile=outfile,
                   voxel_size=voxel_size,
                   shift_to_center=shift_to_center)
