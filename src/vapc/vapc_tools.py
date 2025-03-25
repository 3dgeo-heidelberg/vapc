from .vapc import Vapc
from .datahandler import DataHandler


def initiate_vapc(lazfile, voxel_size, origin=None):
    """
    Initializes a Vapc instance with the provided LAS/LAZ file and voxel parameters.

    This function creates a `DATA_HANDLER` instance to load the LAS/LAZ files,
    initializes a `Vapc` instance with the specified voxel size and origin,
    and associates the data handler with the Vapc instance.

    Parameters
    ----------
    lazfile : str or list of str
        Path to a single LAS/LAZ file or a list of paths to LAS/LAZ files to be processed.
    voxel_size : float
        Defines the size of each voxel for processing.
    origin : list of float, optional
        The origin coordinates [X, Y, Z] for voxelization. Defaults to [0, 0, 0].

    Returns
    -------
    tuple
        A tuple containing:
        - vapc_pc (Vapc): The initialized Vapc instance.
        - dh (DATA_HANDLER): The data handler instance containing the loaded data.
    """
    if origin is None:
        origin = [0, 0, 0]
    dh = DataHandler(lazfile)
    dh.load_las_files()
    vapc_pc = Vapc(float(voxel_size), origin)
    vapc_pc.get_data_from_data_handler(dh)
    return vapc_pc, dh


def mask(vapc_pc, maskfile, segment_in_or_out, buffer_size, reduce_to=False):
    """
    Applies a spatial mask to the Vapc point cloud data.

    This function computes reduction points, loads the mask file, applies buffering,
    and filters the main Vapc point cloud based on the mask criteria.

    Parameters
    ----------
    vapc_pc : Vapc
        The Vapc instance containing the main point cloud data.
    maskfile : str
        Path to the LAS/LAZ file used as a mask.
    segment_in_or_out : str
        Determines whether to keep ("in") or remove ("out") points that overlap with the mask.
        Must be either "in" or "out".
    buffer_size : int
        The size of the buffer to apply around the mask voxels.
    reduce_to : str or bool, optional
        Specifies the method to reduce the DataFrame to one value per voxel after masking.
        Options include "closest_to_center_of_gravity", "center_of_voxel", "center_of_gravity".
        If False, no reduction is performed (keeping all points). Defaults to False.

    Returns
    -------
    pandas.DataFrame
        The filtered (and optionally reduced) DataFrame after applying the mask.

    Examples
    --------
    >>> filtered_df = mask(vapc_pc, "path_to/mask.laz", "in", buffer_size=1, reduce_to="center_of_gravity")
    >>> print(filtered_df.head())
    """
    # Apply offset
    vapc_pc.compute_reduction_point()
    dh_mask = DataHandler(maskfile)
    dh_mask.load_las_files()
    vapc_mask = Vapc(vapc_pc.voxel_size, [0, 0, 0])
    vapc_mask.get_data_from_data_handler(dh_mask)
    # Apply offset
    vapc_mask.compute_reduction_point()
    min_reduction_point = (
        min([vapc_pc.reduction_point[0], vapc_mask.reduction_point[0]]),
        min([vapc_pc.reduction_point[1], vapc_mask.reduction_point[1]]),
        min([vapc_pc.reduction_point[2], vapc_mask.reduction_point[2]]),
    )
    # print(min_reduction_point)
    vapc_pc.reduction_point = min_reduction_point
    vapc_pc.compute_offset()
    vapc_mask.reduction_point = min_reduction_point
    vapc_mask.compute_offset()
    # Buffer mask voxelized point cloud
    vapc_mask.compute_voxel_buffer(buffer_size=int(buffer_size))
    vapc_mask.voxel_index = False
    vapc_mask.df = vapc_mask.buffer_df
    # Select by mask
    vapc_pc.select_by_mask(
        vapc_mask, segment_in_or_out=segment_in_or_out
    )
    # Undo offset
    vapc_pc.compute_offset()
    if reduce_to:  # check if it should be reduced to voxels
        vapc_pc.return_at = reduce_to
        vapc_pc.reduce_to_voxels()
    return vapc_pc.df


def subsample(vapc_pc, reduce_to="closest_to_center_of_gravity"):
    """
    Subsamples the Vapc point cloud data by reducing it to one point per voxel.

    This function sets the reduction method and performs the voxel reduction on the Vapc instance.

    Parameters
    ----------
    vapc_pc : Vapc
        The Vapc instance containing the point cloud data to be subsampled.
    reduce_to : str, optional
        Specifies the method to reduce the DataFrame to one value per voxel.
        Options include "closest_to_center_of_gravity", "center_of_voxel", "center_of_gravity".
        Defaults to "closest_to_center_of_gravity".

    Returns
    -------
    pandas.DataFrame
        The subsampled DataFrame containing one point per voxel.

    Examples
    --------
    >>> subsampled_df = subsample(vapc_pc, reduce_to="center_of_voxel")
    >>> print(subsampled_df.head())
    """
    vapc_pc.return_at = reduce_to
    vapc_pc.reduce_to_voxels()
    return vapc_pc.df


def compute_attributes(vapc_pc, compute, reduce_to="closest_to_center_of_gravity"):
    """
    Computes specified attributes for the Vapc point cloud data.

    This function sets the attributes to be computed, performs the computations,
    and optionally reduces the DataFrame to one value per voxel.

    Parameters
    ----------
    vapc_pc : Vapc
        The Vapc instance containing the point cloud data.
    compute : list of str
        List of attribute names to compute (e.g., ["point_count", "center_of_gravity"]).
    reduce_to : str or bool
        Specifies the method to reduce the DataFrame to one value per voxel after computing attributes.
        Options include "closest_to_center_of_gravity", "center_of_voxel", "corner_of_voxel".
        If False, no reduction is performed.
        Defaults to "closest_to_center_of_gravity".

    Returns
    -------
    pandas.DataFrame
        The DataFrame with the newly computed attributes (and optionally reduced).

    Examples
    --------
    >>> computed_df = compute_attributes(vapc_pc, compute=["point_count", "eigenvalues"], reduce_to="center_of_gravity")
    >>> print(computed_df.head())
    """
    vapc_pc.compute = compute
    vapc_pc.compute_requested_attributes()
    if reduce_to:  # check if it should be reduced to voxels
        vapc_pc.return_at = reduce_to
        vapc_pc.reduce_to_voxels()
    return vapc_pc.df


def filter_by_attributes(vapc_pc, filters, reduce_to="closest_to_center_of_gravity"):
    """
    Filters the Vapc point cloud data based on specified attribute conditions.

    This function computes the necessary attributes, applies the filter conditions,
    and optionally reduces the DataFrame to one value per voxel after filtering.
    Choice of operators: ['equal_to', 'greater_than', 'less_than', 'greater_than_or_equal_to', 'less_than_or_equal_to', '==', '>', '<', '>=', '<=']
    Parameters
    ----------
    vapc_pc : Vapc
        The Vapc instance containing the point cloud data.
    filters : dict
        A dictionary where keys are attribute names and values are dictionaries
        of filter conditions and their corresponding values. Example:
        {
            "point_count": {"greater_than_or_equal_to": 10},
            "eigenvalue_1": {"less_than": .2}
        }
    reduce_to : str or bool
        Specifies the method to reduce the DataFrame to one value per voxel after filtering.
        Options include "closest_to_center_of_gravity", "center_of_voxel", "center_of_gravity".
        If False, no reduction is performed.
        Defaults to "closest_to_center_of_gravity".

    Returns
    -------
    pandas.DataFrame
        The filtered (and optionally reduced) DataFrame.

    Examples
    --------
    >>> filters = {
    ...     "point_count": {"greater_than_or_equal_to": 10},
    ...     "eigenvalue_1": {"less_than": .2}
    ... }
    >>> filtered_df = filter_by_attributes(vapc_pc, filters, reduce_to="center_of_gravity")
    >>> print(filtered_df.head())
    """
    # Compute filter attributes
    vapc_pc.compute = list(filters.keys())
    vapc_pc.compute_requested_attributes()
    # Filter attribute
    for fa in filters.keys():
        filter_attribute = filters[fa]
        for filter_condition in filter_attribute.keys():
            if filter_condition in ['equal_to', 'greater_than', 'less_than', 'greater_than_or_equal_to', 'less_than_or_equal_to', '==', '>', '<', '>=', '<=']:
                vapc_pc.filter_attributes(
                    fa,
                    filter_condition,
                    filter_attribute[filter_condition],
                )
            else:
                return False
    if reduce_to:  # check if it should be reduced to voxels
        vapc_pc.return_at = reduce_to
        vapc_pc.reduce_to_voxels()
    return vapc_pc.df


def filter_by_attributes_and_compute(vapc_pc, filters, compute, reduce_to):
    """
    Filters the Vapc point cloud data based on attributes and then computes additional attributes.

    This function first applies attribute-based filtering and then computes specified attributes
    on the filtered data. It ensures that there are no redundant computations for attributes
    already used in filtering.

    Parameters
    ----------
    vapc_pc : Vapc
        The Vapc instance containing the point cloud data.
    filters : dict
        A dictionary where keys are attribute names and values are dictionaries
        of filter conditions and their corresponding values. Example:
        {
            "point_count": {"greater_equal": 10},
            "eigenvalue_1": {"less": .5}
        }
    compute : list of str
        List of additional attribute names to compute after filtering.
    reduce_to : str or bool
        Specifies the method to reduce the DataFrame to one value per voxel after filtering and computing.
        Options include "closest_to_center_of_gravity", "center_of_voxel", "corner_of_voxel".
        If False, no reduction is performed.
        Defaults to "closest_to_center_of_gravity".

    Returns
    -------
    pandas.DataFrame
        The filtered and computed (and optionally reduced) DataFrame.

    Examples
    --------
    >>> filters = {
    ...     "point_count": {"greater_equal": 10}
    ... }
    >>> compute = ["eigenvalues", "covariance_matrix"]
    >>> final_df = filter_by_attributes_and_compute(vapc_pc, filters, compute, reduce_to="center_of_gravity")
    >>> print(final_df.head())
    """
    # Lets filter first
    vapc_pc.df = filter_by_attributes(vapc_pc, filters, reduce_to=False)

    # Prevent multiple computations of the same attribute:
    for filter_attr in filters.keys():
        compute = list(filter((filter_attr).__ne__, compute))

    # And compute attributes after
    return compute_attributes(vapc_pc, compute, reduce_to)


def compute_statistics(vapc_pc, statistics, reduce_to="closest_to_center_of_gravity"):
    """
    Computes statistical attributes for the Vapc point cloud data.

    This function sets the statistical attributes to be computed, performs the computations,
    and optionally reduces the DataFrame to one value per voxel.

    Parameters
    ----------
    vapc_pc : Vapc
        The Vapc instance containing the point cloud data.
    statistics : dict
        A dictionary of statistical attributes to compute. Example:
        {
            "point_count": "mean",
            "eigenvalues": ["mean", "std"]
        }
    reduce_to : str or bool
        Specifies the method to reduce the DataFrame to one value per voxel after computing statistics.
        Options include "closest_to_center_of_gravity", "center_of_voxel", "center_of_gravity".
        If False, no reduction is performed.
        Defaults to "closest_to_center_of_gravity".

    Returns
    -------
    pandas.DataFrame
        The DataFrame with the computed statistical attributes (and optionally reduced).

    Examples
    --------
    >>> statistics = {
    ...     "point_count": "mean",
    ...     "eigenvalues": ["mean", "std"]
    ... }
    >>> stats_df = compute_statistics(vapc_pc, statistics, reduce_to="center_of_gravity")
    >>> print(stats_df.head())
    """
    vapc_pc.attributes = statistics
    vapc_pc.compute_requested_statistics_per_attributes()
    if reduce_to:  # check if it should be reduced to voxels
        vapc_pc.return_at = reduce_to
        vapc_pc.reduce_to_voxels()
    return vapc_pc.df


def use_tool(tool_name, infile, outfile, voxel_size, args, reduce_to):
    """
    Executes a specified Vapc tool on the input file and saves the output.

    This function initializes the Vapc and DATA_HANDLER instances, applies the chosen tool,
    and saves the processed data in the desired output format.

    Parameters
    ----------
    tool_name : str
        The name of the Vapc tool to execute. Options include:
        "subsample", "mask", "compute", "filter", "filter_and_compute", "statistics".
    infile : str
        Path to the input LAS/LAZ file.
    outfile : str
        Path where the output file will be saved.
    voxel_size : float
        Defines the size of each voxel for processing.
    args : dict
        A dictionary of arguments specific to the chosen tool.
        The expected keys vary depending on `tool_name`.
    reduce_to : str or bool
        Specifies the method to reduce the DataFrame to one value per voxel after processing.
        Options include "closest_to_center_of_gravity", "center_of_voxel", "corner_of_voxel".
        If False, no reduction is performed.

    Returns
    -------
    None or str
        Returns None if the tool executes successfully.
        Returns an error message string if an unknown tool is specified.

    Raises
    ------
    KeyError
        If required keys are missing in the `args` dictionary.
    ValueError
        If an unknown `tool_name` is provided.

    Examples
    --------
    >>> args = {
    ...     "sub_sample_method": "center_of_gravity"
    ... }
    >>> use_tool("subsample", "data/input.laz", "data/output.laz", 0.5, args, "center_of_gravity")
    >>> print(os.path.exists("data/output.laz"))
    """
    vapc_pc, dh = initiate_vapc(infile, voxel_size, origin=[0, 0, 0])

    if tool_name == "subsample":
        dh.df = subsample(vapc_pc=vapc_pc, reduce_to=args["sub_sample_method"])

    elif tool_name == "mask":
        dh.df = mask(
            vapc_pc=vapc_pc,
            maskfile=args["maskfile"],
            segment_in_or_out=args["segment_in_or_out"],
            buffer_size=args["buffer_size"],
            reduce_to=reduce_to,
        )

    elif tool_name == "compute":
        dh.df = compute_attributes(
            vapc_pc=vapc_pc, compute=args["compute"], reduce_to=reduce_to
        )

    elif tool_name == "filter":
        dh.df = filter_by_attributes(
            vapc_pc=vapc_pc, filters=args["filters"], reduce_to=reduce_to
        )

    elif tool_name == "filter_and_compute":
        dh.df = filter_by_attributes_and_compute(
            vapc_pc=vapc_pc,
            filters=args["filters"],
            compute=args["compute"],
            reduce_to=reduce_to,
        )
    elif tool_name == "statistics":
        dh.df = compute_statistics(
            vapc_pc=vapc_pc, statistics=args["statistics"], reduce_to=reduce_to
        )
    else:
        raise ValueError(f"unknown tool '{tool_name}'")
    if outfile.endswith(".las") or outfile.endswith(".las"):
        dh.save_as_las(outfile)
    elif outfile.endswith(".ply"):
        print("PLY")
        dh.save_as_ply(outfile, voxel_size, shift_to_center=False)
    else:
        return "Unknown output format"

def lasz_to_ply(infile, outfile, voxel_size, shift_to_center=False):
    """
    Converts a LAS/LAZ file to PLY format.

    This function loads the LAS/LAZ file using `DATA_HANDLER` and saves it as a PLY file
    with the specified voxel size and optional shifting to the voxel center.

    Parameters
    ----------
    infile : str
        Path to the input LAS/LAZ file.
    outfile : str
        Path where the output PLY file will be saved.
    voxel_size : float
        Defines the size of each voxel for processing.
    shift_to_center : bool, optional
        If True, shifts the points to the origin. Defaults to False.

    Returns
    -------
    True

    Examples
    --------
    >>> lasz_to_ply("data/input.laz", "data/output.ply", 0.5, shift_to_center=True)
    >>> print(os.path.exists("data/output.ply"))
    """
    dh = DataHandler(infile)
    dh.load_las_files()
    dh.save_as_ply(
        outfile=outfile, voxel_size=voxel_size, shift_to_center=shift_to_center
    )
    return True




########################## MESH-Voxel-PC tools ##################################
def mesh_vertices_to_vapc( mesh_file,
                          skip_rows = 2, 
                          voxel_size = 1):
        dh = DataHandler(mesh_file)
        dh.open_obj_mesh(skiprows=skip_rows)
        vp = Vapc(voxel_size)
        vp.df = dh.vertex_df
        return vp

def point_cloud_to_vapc(point_cloud_file, 
                        voxel_size = 1):
        dh = DataHandler(point_cloud_file)
        dh.load_las_files()
        vp = Vapc(voxel_size)
        vp.df = dh.df
        return vp

def select_mesh_by_mask(dh_scene,vp_scene,outfile,strict=True): #TODO: Add notebook example
    """
    Selects and saves a mesh based on a vertex mask.

    This function filters the faces of a mesh based on whether their vertices
    are included in a given mask. The filtered mesh is then saved to an OBJ file.

    Parameters:
    dh_scene (object): An object representing the scene containing the mesh to be filtered.
                        It should have a 'face_df' attribute which is a DataFrame containing
                        the faces of the mesh, with columns 'v1', 'v2', and 'v3' representing
                        the vertex IDs of each face.
    vp_scene (object): A Vapc object representing the scene containing the vertex mask.
                        It should have a 'df' attribute which is a DataFrame containing
                        the vertex IDs in a column named 'vertex_id'.
    outfile (str): The path to the output file where the filtered mesh will be saved.
    strict (bool, optional): If True, only faces where all vertices are in the mask
                                will be selected. If False, faces where at least one
                                vertex is in the mask will be selected. Default is True.

    Returns:
    None
    """
    # Make a working copy of the faces DataFrame.
    # Get the list (or Series) of vertex IDs from the mask.
    mask_vertex_ids = vp_scene.df['vertex_id']
    # Create boolean Series indicating whether each face vertex is in the mask.
    v1_in = dh_scene.face_df['v1'].isin(mask_vertex_ids)
    v2_in = dh_scene.face_df['v2'].isin(mask_vertex_ids)
    v3_in = dh_scene.face_df['v3'].isin(mask_vertex_ids)
    # Determine the selection mask based on the segmentation mode and strictness.
    if strict:
        # Strict: all vertices must be inside.
        selection = v1_in & v2_in & v3_in
    else:
        # Non-strict: at least one vertex inside.
        selection = v1_in | v2_in | v3_in
    # Filter the face DataFrame.
    dh_scene.face_df = dh_scene.face_df[selection]
    # Save the segmented OBJ mesh.
    dh_scene.save_obj_mesh(outfile)

def extract_point_cloud_by_3D_mask(scene_file, mask_file, outfile, voxel_size = 1, segment_mode='in', mode = "p", skiprows = 2): #TODO: Add notebook example
    """
    Extracts a point cloud from a scene file using a 3D mask and saves the result to an output file.

    Parameters:
    scene_file (str): Path to the input scene file containing the point cloud data.
    mask_file (str): Path to the mask file used to filter the point cloud.
    outfile (str): Path to the output file where the filtered point cloud will be saved.
    voxel_size (int, optional): Size of the voxel grid used for processing. Default is 1.
    segment_mode (str, optional): Mode for segmenting the point cloud. Can be 'in' or 'out'. Default is 'in'.
    mode (str, optional): Mode for processing the mask file. 'm' for mesh vertices, 'p' for point cloud. Default is 'p'.
    skiprows (int, optional): Number of rows to skip when reading the mask file. Default is 2.

    Raises:
    ValueError: If the mode is not 'm' or 'p'.

    Returns:
    None
    """
    dh_scene = DataHandler(scene_file)
    dh_scene.load_las_files()
    if mode == "m":
        vp_mask = mesh_vertices_to_vapc(mask_file,skip_rows = skiprows, voxel_size = voxel_size)
    elif mode == "p":
        vp_mask = point_cloud_to_vapc(mask_file, voxel_size = voxel_size)
    else:   
        raise ValueError("mode must be either 'm' or 'p', seperated by '_'")
    vp_scene = Vapc(voxel_size)
    vp_scene.df = dh_scene.df
    # Use vapc select_by_mask method to select vertices inside the mask.
    vp_scene.select_by_mask(vp_mask,segment_in_or_out=segment_mode)
    dh_scene.df = vp_scene.df
    dh_scene.save_as_las(outfile)
                   
def extract_mesh_by_3D_mask(scene_file, mask_file, outfile, skiprows = 2, skiprows_mask = 2, voxel_size = 1, mode = "m", segment_mode='in', strict=True): #TODO: Add notebook example
    """
    Extracts a mesh from a 3D scene file using a 3D mask and saves the result to an output file.

    Parameters:
    scene_file (str): Path to the 3D scene file.
    mask_file (str): Path to the 3D mask file.
    outfile (str): Path to the output file where the extracted mesh will be saved.
    skiprows (int, optional): Number of rows to skip when reading the scene file. Default is 2.
    skiprows_mask (int, optional): Number of rows to skip when reading the mask file. Default is 2.
    voxel_size (int, optional): Size of the voxel for the mask. Default is 1.
    mode (str, optional): Mode for processing the mask file. 'm' for mesh vertices, 'p' for point cloud. Default is 'm'.
    segment_mode (str, optional): Mode for segmenting the mesh. 'in' to select vertices inside the mask, 'out' to select vertices outside the mask. Default is 'in'.
    strict (bool, optional): If True, only faces where all vertices are in the mask will be selected. If False, faces where at least one vertex is in the mask will be selected. Default is True.

    Returns:
    bool: True if the mesh extraction is successful.
    
    Raises:
    ValueError: If the mode is not 'm' or 'p'.
    """
    dh_scene = DataHandler(scene_file)
    dh_scene.open_obj_mesh(skiprows=skiprows)
    if mode == "m":
        vp_mask = mesh_vertices_to_vapc(mask_file,skip_rows = skiprows_mask, voxel_size = voxel_size)
    elif mode == "p":
        vp_mask = point_cloud_to_vapc(mask_file, voxel_size = voxel_size)
    else:   
        raise ValueError("mode must be either 'm' or 'p', seperated by '_'")
    vp_scene = Vapc(voxel_size)
    vp_scene.df = dh_scene.vertex_df
    # Use vapc select_by_mask method to select vertices inside the mask.
    vp_scene.select_by_mask(vp_mask,segment_in_or_out=segment_mode)
    select_mesh_by_mask(dh_scene,vp_scene,outfile,strict=strict)
    return True                          


def extract_areas_with_change_using_mahalanobis_distance(point_cloud_1_path, point_cloud_2_path, mask_file, point_cloud_out_1_path, point_cloud_out_2_path, voxel_size, alpha_value, delete_mask_file=True):
    #Open point clouds
    vapc_1 = point_cloud_to_vapc(point_cloud_file=point_cloud_1_path,
                            voxel_size=voxel_size)
    vapc_2 = point_cloud_to_vapc(point_cloud_file=point_cloud_2_path,
                            voxel_size=voxel_size)

    # Initiate Bi-temporal VAPC
    bi_vapc = vapc.BiTemporalVapc([vapc_1, vapc_2])
    bi_vapc.prepare_data_for_mahalanobis_distance() #prepare data for mahalanobis distance
    bi_vapc.merge_vapcs_with_same_voxel_index() #defines how comparison is done. Here, it is done per same voxel index.
    # bi_vapc.compute_distance() #euclidean distance -> optional
    bi_vapc.compute_mahalanobis_distance(alpha=alpha_value) #alpha value for chi2
    bi_vapc.compute_voxels_occupied_in_single_epoch() #Also get disappearing and appearing voxels

    # Prepare data for export
    bi_vapc.prepare_data_for_export()
    # As we only want areas where change might have happened areas where the change 
    # type is 1 (mahalanobis significant), 2 (less than 30 points in voxel), 3 (disappearing), 
    # and 4 (appearing) are kept.
    bi_vapc.df = bi_vapc.df[(bi_vapc.df["change_type"] >= 1)]
    bi_vapc.save_to_las(mask_file)

    # Clip area from T1 and T2
    extract_point_cloud_by_3D_mask(point_cloud_1_path, mask_file, point_cloud_out_1_path, voxel_size)
    extract_point_cloud_by_3D_mask(point_cloud_2_path, mask_file, point_cloud_out_2_path, voxel_size)

    if delete_mask_file:
        os.remove(mask_file)