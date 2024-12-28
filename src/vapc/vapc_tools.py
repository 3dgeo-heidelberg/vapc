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
    vapc_mask.df = vapc_mask.buffer_df
    # Select by mask
    vapc_pc.select_by_mask(
        vapc_mask, mask_attribute="voxel_index", segment_in_or_out=segment_in_or_out
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
        return "unknown command:%s" % tool_name
    dh.save_as_las(outfile)


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
