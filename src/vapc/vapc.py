import math
import numpy as np
from scipy.spatial import KDTree
import pandas as pd
from .utilities import trace, timeit, compute_mode_continuous


class Vapc:
    # TODO: @Ronny: Why AVAILABE_COMPUTATIONS as class attribute and as instance attribute?
    AVAILABLE_COMPUTATIONS = [
        "voxel_index",
        "point_count",
        "point_density",
        "percentage_occupied",
        "covariance_matrix",
        "eigenvalues",
        "geometric_features",
        "center_of_gravity",
        "distance_to_center_of_gravity",
        "std_of_cog",
        "closest_to_center_of_gravity",
        "center_of_voxel",
        "corner_of_voxel",
    ]

    def __init__(
        self,
        voxel_size: float,
        origin: list = None,
        attributes: dict = None,
        compute: list = None,
        return_at: str = "closest_to_center_of_gravity",
    ):
        """
        Initializes the Vapc (Voxel Analysis for Point clouds) class.

        Parameters
        ----------
        voxel_size : float
            Defines the size of each voxel.
        origin : list, optional
            Defines the origin of the voxel space. Defaults to [0,0,0].
        attributes : dict, optional
            Dictionary containing information about which attributes to read and
            what statistics to compute on them. Defaults to an empty dictionary.
        compute : list, optional
            List containing names of attributes that will be calculated when
            calling 'compute_requested_attributes'. Defaults to an empty list.
        return_at : str, optional
            Specifies the point to which the data will be reduced when calling
            'reduce_to_voxels'. Determines the location of each output voxel.
            Defaults to "closest_to_center_of_gravity".

        Notes
        -----
        The class is under construction and may be subject to changes.
        """
        # Relevant input:
        self.voxel_size = voxel_size
        self.origin = origin if origin is not None else [0, 0, 0]
        self.attributes = attributes if attributes is not None else {}
        self.compute = compute if compute is not None else []
        self.return_at = return_at

        # Validate some parameters
        if self.voxel_size <= 0:  # check voxel size
            raise ValueError("voxel_size must be a positive number.")
        if not isinstance(self.origin, list) or len(self.origin) != 3:  # check origin
            raise ValueError("origin must be a list of three coordinates [x, y, z].")

        self.AVAILABLE_COMPUTATIONS = [
            "voxel_index",
            "point_count",
            "point_density",
            "percentage_occupied",
            "covariance_matrix",
            "eigenvalues",
            "geometric_features",
            "center_of_gravity",
            "distance_to_center_of_gravity",
            "std_of_cog",
            "closest_to_center_of_gravity",
            "center_of_voxel",
            "corner_of_voxel",
        ]
        self.df = None
        self.original_attributes = None

        # Calculations not applied yet:
        self.attributes_up_to_data = False
        self.voxelized = False
        self.voxel_index = False
        self.point_count = False
        self.point_density = False
        self.percentage_occupied = False
        self.covariance_matrix = False
        self.eigenvalues = False
        self.geometric_features = False
        self.center_of_gravity = False
        self.distance_to_center_of_gravity = False
        self.std_of_cog = False
        self.closest_to_center_of_gravity = False
        self.center_of_voxel = False
        self.corner_of_voxel = False
        self.attributes_per_voxel = False
        self.drop_columns = []
        self.new_column_names = {}
        self.reduced = False
        self.reduction_point = None
        self.offset_applied = False

    def _validate_compute_list(self):
        """
        Validates the 'compute' list to ensure all computations are valid.

        Raises
        ------
        ValueError
            If any computation in 'self.compute' is not in 'self.AVAILABLE_COMPUTATIONS'.
        """
        invalid_computations = [
            comp for comp in self.compute if comp not in self.AVAILABLE_COMPUTATIONS
        ]
        if invalid_computations:
            raise ValueError(
                f"Invalid computation(s) requested: {invalid_computations}. "
                f"Available computations are: {self.AVAILABLE_COMPUTATIONS}"
            )

    @trace
    @timeit
    def get_data_from_data_handler(self, data_handler):
        """
        Moves the dataframe from the data handler to the Vapc instance.

        After this operation, data_handler will no longer have the 'df' attribute.

        Parameters
        ----------
        data_handler : DATA_HANDLER
            An instance of the DATA_HANDLER class from which to move the dataframe.
        """
        if data_handler.df is None:
            raise AttributeError(
                "The provided data_handler does not have a 'df' attribute."
            )
        self.df = data_handler.df
        data_handler.df = None  # Remove df from data_handler
        self.original_attributes = self.df.columns.tolist()

    @trace
    @timeit
    def compute_reduction_point(self):
        """
        Computes the minimum 'X', 'Y', and 'Z' values in the dataset and stores them in `self.reduction_point`.
        This point can be used as a reference or offset for further computations.
        """
        self.reduction_point = [
            int(self.df["X"].min()),
            int(self.df["Y"].min()),
            int(self.df["Z"].min()),
        ]

    @trace
    @timeit
    def compute_offset(self):
        """
        Applies or removes an offset to the 'X', 'Y', and 'Z' coordinates based on `self.reduction_point`.

        If `self.offset_applied` is False, the offset is subtracted from the coordinates, and `self.offset_applied` is set to True.
        If `self.offset_applied` is True, the offset is added back to the coordinates, and `self.offset_applied` is set to False.
        """
        if self.reduction_point is None:
            self.compute_reduction_point()
        if self.offset_applied:
            self.df["X"] += self.reduction_point[0]
            self.df["Y"] += self.reduction_point[1]
            self.df["Z"] += self.reduction_point[2]
            self.offset_applied = False
        else:
            self.df["X"] -= self.reduction_point[0]
            self.df["Y"] -= self.reduction_point[1]
            self.df["Z"] -= self.reduction_point[2]
            self.offset_applied = True

    @trace
    @timeit
    def voxelize(self):
        """
        Computes voxel indices for each point and adds them as new columns in `self.df`.

        The voxel indices are calculated by subtracting `self.origin` from the point coordinates,
        dividing by `self.voxel_size`, and taking the floor of the result.

        Adds the following columns to `self.df`:
            - 'voxel_x', 'voxel_y', 'voxel_z': The voxel indices along each axis.
        """
        for i, dim in enumerate(["X", "Y", "Z"]):
            self.df[f"voxel_{dim.lower()}"] = np.floor(
                (self.df[dim] - self.origin[i]) / self.voxel_size
            ).astype(int)

        # Optionally create a unique voxel ID by combining voxel indices
        # Uncomment the following line if you need a voxel identifier
        # self.df["voxel_id"] = self.df[['voxel_x', 'voxel_y', 'voxel_z']].astype(str).agg('_'.join, axis=1)

        self.voxelized = True

    @trace
    @timeit
    def compute_requested_attributes(self):
        """
        Computes attributes based on the 'compute' input list.

        This method iterates over the list of computations specified in `self.compute` and calls
        the corresponding methods to compute various attributes of the voxel data.

        The available computations are:
            - "voxel_index"
            - "point_count"
            - "point_density"
            - "percentage_occupied"
            - "covariance_matrix"
            - "eigenvalues"
            - "geometric_features"
            - "center_of_gravity"
            - "distance_to_center_of_gravity"
            - "std_of_cog"
            - "closest_to_center_of_gravity"
            - "center_of_voxel"
            - "corner_of_voxel"
        """

        # Validate the 'compute' list
        self._validate_compute_list()

        for computation_name in self.compute:
            if getattr(self, computation_name) is True:
                continue
            method_name = f"compute_{computation_name}"
            method = getattr(self, method_name, None)
            if callable(method):
                try:
                    # print(f"Starting computation of '{computation_name}'")
                    method()
                    # print(f"Successfully computed '{computation_name}'")
                except Exception as e:
                    print(f"Error computing '{computation_name}': {e}")
                    raise e
            else:
                print(f"Invalid computation name: '{computation_name}'")
                raise ValueError(f"Invalid computation name: '{computation_name}'")

    def update_attribute_dictionary(self, remove_cols=None):
        """
        Updates the attribute dictionary with default statistics.

        If not specified, the mean will be computed for each attribute when the point cloud is voxelized.
        If a specific statistic is specified for an attribute, it will be used instead.

        Parameters
        ----------
        remove_cols : list, optional
            List of column names to remove from the attributes to be processed.
            Defaults to ['X', 'Y', 'Z', 'bit_fields', 'raw_classification',
            'scan_angle_rank', 'user_data', 'point_source_id'].

        Notes
        -----
        - The method updates `self.attributes` by setting the default statistic to "mean" for each attribute,
        unless a different statistic is already specified in `self.attributes`.
        - Sets `self.attributes_up_to_data` to True after updating the attributes.
        """
        if remove_cols is None:
            remove_cols = [
                "X",
                "Y",
                "Z",
                "bit_fields",
                "raw_classification",
                "scan_angle_rank",
                "user_data",
                "point_source_id",
            ]

        original_attributes = list(self.original_attributes)
        for dc in remove_cols:
            try:
                original_attributes.remove(dc)
            except:
                pass

        attributes = {}
        for col in original_attributes:
            attributes[col] = "mean"
        for attr in self.attributes.keys():
            attributes[attr] = self.attributes[attr]
        self.attributes = attributes
        self.attributes_up_to_data = True

    @trace
    @timeit
    def compute_requested_statistics_per_attributes(self):
        """
        Computes the requested statistics per voxel for specified attributes using NumPy.

        This method computes various statistics (e.g., mean, median, mode) for each attribute in `self.attributes`,
        grouped by voxel indices ('voxel_x', 'voxel_y', 'voxel_z').

        The available statistics are:
            - "mean"
            - "std"
            - "var"
            - "median"
            - "mode"
            - "min"
            - "max"
            - "sum"
            - "mode_count"

        For "mode_count", an additional parameter can be specified to define the percentage threshold.

        Notes:
            - If the data is not voxelized (`self.voxelized` is False), the method will voxelize the data first.
            - If the attribute dictionary is not updated (`self.attributes_up_to_data` is False), it will be updated.

        Raises:
            KeyError: If required columns are missing in `self.df`.
            ValueError: If invalid statistics are specified.

        Side Effects:
            - Updates `self.df` by merging the computed statistics.
            - Sets `self.attributes_per_voxel` to True.
        """
        
        #Check if any attributes are specified
        if self.attributes == {}:
            print("No attributes specified for computation. Skipping.")
            return

        # Ensure data is voxelized
        if not self.voxelized:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]

        # Update attribute dictionary if needed
        if not self.attributes_up_to_data:
            self.update_attribute_dictionary()

        # Check required columns
        required_columns = ["voxel_x", "voxel_y", "voxel_z"] + list(
            self.attributes.keys()
        )
        missing_columns = [
            col for col in required_columns if col not in self.df.columns
        ]
        if missing_columns:
            raise KeyError(f"Missing columns in `self.df`: {missing_columns}")

        # Subset the DataFrame
        df_temp_subset = self.df[required_columns]

        # Convert DataFrame to structured NumPy array
        dtypes = [(col, df_temp_subset[col].dtype) for col in df_temp_subset.columns]
        data = np.array([tuple(row) for row in df_temp_subset.values], dtype=dtypes)

        # Sort data by voxel indices
        sorted_indices = np.lexsort((data["voxel_z"], data["voxel_y"], data["voxel_x"]))
        sorted_data = data[sorted_indices]

        # Find unique voxel groups and their indices
        voxel_keys = sorted_data[["voxel_x", "voxel_y", "voxel_z"]]
        groups, indices = np.unique(voxel_keys, return_index=True)

        # Prepare to collect aggregated data
        aggregated_data = {}
        attribute_names = []
        # print("Computing statistics per voxel...")

        for attr, stats_list in self.attributes.items():
            if not isinstance(stats_list, list):
                stats_list = [stats_list]
            for stat in stats_list:
                # print(f"Computing '{stat}' for attribute '{attr}'")
                # start_time = time.time()
                aggregated_values = []
                for i in range(len(indices)):
                    start_idx = indices[i]
                    end_idx = (
                        indices[i + 1] if i + 1 < len(indices) else len(sorted_data)
                    )
                    group_slice = sorted_data[attr][start_idx:end_idx]
                    if stat == "mean":
                        value = group_slice.mean()
                    elif stat == "std":
                        value = group_slice.std()
                    elif stat == "var":
                        value = group_slice.var()
                    elif stat == "median":
                        value = np.median(group_slice)
                    elif stat == "min":
                        value = group_slice.min()
                    elif stat == "max":
                        value = group_slice.max()
                    elif stat == "sum":
                        value = group_slice.sum()
                    elif stat == "mode":
                        try:
                            counts = np.bincount(group_slice.astype(int))
                            value = np.argmax(counts)
                        except ValueError:
                            value = compute_mode_continuous(group_slice)
                    elif "mode_count" in stat:
                        # Extract percentage threshold
                        try:
                            _, percentage_str = stat.split(",")
                            percentage = float(percentage_str)
                        except ValueError as exc:
                            raise ValueError(
                                f"Invalid 'mode_count' specification for attribute '{attr}': '{stat}'"
                            ) from exc
                        try:
                            counts = np.bincount(group_slice.astype(int))
                            counts_sum = counts.sum()
                        except ValueError as exc:
                            raise ValueError(
                                f"Cannot compute 'mode_count' for continuous attribute '{attr}'"
                            ) from exc
                    
                        counts = counts[counts!=0]
                        proportions = counts / counts_sum
                        count_above_threshold = np.sum(proportions >= percentage)
                        value = count_above_threshold
                    else:
                        print(
                            f"Unknown aggregation type '{stat}' for attribute '{attr}'. Skipping."
                        )
                        continue
                    aggregated_values.append(value)
                # end_time = time.time()
                # print(f"Computed '{stat}' for attribute '{attr}' in {end_time - start_time:.2f} seconds")
                col_name = f"{attr}_{stat}"
                aggregated_data[col_name] = aggregated_values
                attribute_names.append(col_name)

        # Build the result array
        result_dtype = [
            ("voxel_x", sorted_data["voxel_x"].dtype),
            ("voxel_y", sorted_data["voxel_y"].dtype),
            ("voxel_z", sorted_data["voxel_z"].dtype),
        ]
        result_dtype += [
            (name, np.array(values).dtype) for name, values in aggregated_data.items()
        ]

        result_array = np.zeros(len(groups), dtype=result_dtype)
        result_array["voxel_x"] = groups["voxel_x"]
        result_array["voxel_y"] = groups["voxel_y"]
        result_array["voxel_z"] = groups["voxel_z"]

        for name, values in aggregated_data.items():
            result_array[name] = values

        # Convert result array to DataFrame
        grouped_df = pd.DataFrame(result_array)

        # Merge aggregated data back into self.df
        self.df = self.df.merge(
            grouped_df, on=["voxel_x", "voxel_y", "voxel_z"], how="left"
        )

        self.attributes_per_voxel = True

    @trace
    @timeit
    def compute_voxel_index(self):
        """
        Computes a unique voxel index using voxel coordinates as a tuple.

        Notes
        -----
        - Adds a new column 'voxel_index' to `self.df`.
        """
        if self.voxel_index:
            return
        if not self.voxelized:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
        self.df.set_index(["voxel_x", "voxel_y", "voxel_z"], inplace=True,drop = False)
        self.df.index.set_names(["idx_voxel_x", "idx_voxel_y", "idx_voxel_z"], inplace=True)
        self.voxel_index = True

    @trace
    @timeit
    def compute_voxel_buffer(self, buffer_size: int = 1):
        """
        Computes a buffer around each voxel by expanding voxel coordinates 
        within a specified buffer size, overwrites self.df with that mask,
        and returns it so you can immediately call select_by_mask().
        """

        # 1) Ensure points have voxel coords
        if not self.voxelized:
            self.voxelize()

        # 2) Pull unique voxel coords from the columns
        coords = (
            self.df[["voxel_x", "voxel_y", "voxel_z"]]
            .drop_duplicates()
            .to_numpy(dtype=int)
        )  # shape = (n_voxels, 3)

        # 3) Build the offset grid
        offsets = np.arange(-buffer_size, buffer_size + 1, dtype=int)
        grid = (
            np.stack(np.meshgrid(offsets, offsets, offsets, indexing="ij"), axis=-1)
            .reshape(-1, 3)
        )  # shape = (n_offsets, 3)

        # 4) Apply every offset to every voxel
        expanded = (coords[:, None, :] + grid[None, :, :]).reshape(-1, 3)

        # 5) Deduplicate into a DataFrame
        buf = pd.DataFrame(expanded, columns=["voxel_x", "voxel_y", "voxel_z"])
        buf = buf.drop_duplicates().reset_index(drop=True)

        # ——— now overwrite self.df with the buffered‐voxel mask ———
        self.df = buf
        self.voxelized = True
        self.voxel_index = False   # force recompute
        # build the MultiIndex so select_by_mask can do df.index.isin(...)
        self.compute_voxel_index()  # sets self.df.index to (voxel_x,voxel_y,voxel_z)

        # Adding the voxel 
        self.df['Y'] = (self.df['voxel_y']+self.voxel_size/2)*self.voxel_size
        self.df['Z'] = (self.df['voxel_z']+self.voxel_size/2)*self.voxel_size
        self.df['X'] = (self.df['voxel_x']+self.voxel_size/2)*self.voxel_size

        self.buffer_df = self.df
        
        return self.df
    
    @trace
    @timeit
    def select_by_mask(
        self, vapc_mask, segment_in_or_out="in"
    ):
        """
        Filters the data points based on a mask provided by another Vapc instance.

        This method either keeps or removes points that overlap with the mask, depending on the
        `segment_in_or_out` parameter.

        Parameters
        ----------
        vapc_mask : Vapc
            Another instance of the Vapc class that provides the mask for filtering.
        mask_attribute : str, optional
            The attribute used for masking. Defaults to "voxel_index".
        segment_in_or_out : str, optional
            Determines whether to keep ("in") or remove ("out") the overlapping points.
            Must be either "in" or "out". Defaults to "in".

        Raises
        ------
        ValueError
            If `segment_in_or_out` is not "in" or "out".
        AttributeError
            If required attributes or data are missing.

        Notes
        -----
        - The method modifies `self.df` in-place by filtering data points.
        - Resets `self.voxelized` to False after filtering.
        - Removes voxel columns and `mask_attribute` from `self.df` after filtering.
        """

        if self.df is None or vapc_mask.df is None:
            raise AttributeError("Both `self.df` and `vapc_mask.df` must exist.")

        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]

        if vapc_mask.voxelized is False:
            vapc_mask.voxelize()

        if vapc_mask.voxel_index is False:
            vapc_mask.compute_voxel_index()

        if self.voxel_index is False:
            self.compute_voxel_index()

        mask_values = vapc_mask.df.index
        if segment_in_or_out == "in":
            self.df = self.df.loc[self.df.index.isin(mask_values)]
        elif segment_in_or_out == "out":
            self.df = self.df.loc[~self.df.index.isin(mask_values)]
        else:
            raise ValueError(
                "Parameter 'segment_in_or_out' must be either 'in' or 'out'."
            )

        for attr in ["voxel_x", "voxel_y", "voxel_z"]:
            try:
                self.df = self.df.drop([attr], axis=1)
            except:
                pass
        self.voxelized = False

    
    @trace
    @timeit
    def label_by_mask(
        self, vapc_mask, label_attributes=["voxel_x", "voxel_y", "voxel_z"]):
        """
        Labels the data points based on a mask provided by another Vapc instance.

        This method either keeps or removes points that overlap with the mask, depending on the
        `segment_in_or_out` parameter.

        Parameters
        ----------
        vapc_mask : Vapc
            Another instance of the Vapc class that provides the mask for filtering.
        mask_attribute : str, optional
            The attribute used for masking. Defaults to "voxel_index".
        segment_in_or_out : str, optional
            Determines whether to keep ("in") or remove ("out") the overlapping points.
            Must be either "in" or "out". Defaults to "in".

        Raises
        ------
        ValueError
            If `segment_in_or_out` is not "in" or "out".
        AttributeError
            If required attributes or data are missing.

        Notes
        -----
        - The method modifies `self.df` in-place by filtering data points.
        - Resets `self.voxelized` to False after filtering.
        - Removes voxel columns and `mask_attribute` from `self.df` after filtering.
        """

        if self.df is None or vapc_mask.df is None:
            raise AttributeError("Both `self.df` and `vapc_mask.df` must exist.")

        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]

        if vapc_mask.voxelized is False:
            vapc_mask.voxelize()

        if vapc_mask.voxel_index is False:
            vapc_mask.compute_voxel_index()

        if self.voxel_index is False:
            self.compute_voxel_index()
           

        # Create a list of labels that exist in vapc_mask.df.columns
        common_cols = [label for label in label_attributes if label in vapc_mask.df.columns]

        # Join only those columns into self.df based on matching index values.
        # You can change 'how' to 'inner' if you only want rows with a match in vapc_mask.df.
        if common_cols:
            self.df = self.df.join(vapc_mask.df[common_cols], how='left')


        for attr in ["voxel_x", "voxel_y", "voxel_z"]:
            try:
                self.df = self.df.drop([attr], axis=1)
            except:
                pass
        self.voxelized = False

    @trace
    @timeit
    def compute_point_count_old2(self):
        """
        Computes the point count for all occupied voxels.

        This method calculates the number of points within each voxel and adds a new column 'point_count' to `self.df`.
        """
        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
        grouped = self.df.groupby(["voxel_x", "voxel_y", "voxel_z"])
        points_per_voxel = grouped.size().reset_index(name="point_count")
        self.df = self.df.merge(
            points_per_voxel, how="left", on=["voxel_x", "voxel_y", "voxel_z"]
        )
        self.point_count = True
        
    @trace
    @timeit
    def compute_point_count_old1(self):
        """
        Computes the point count for all occupied voxels.

        This method calculates the number of points within each voxel and adds a new column 'point_count' to `self.df`.
        """
        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
        group_keys = ["voxel_x", "voxel_y", "voxel_z"]
        grouped = self.df.groupby(group_keys)
        self.df["point_count"] = grouped["X"].transform("size")
        self.point_count = True

    @trace
    @timeit
    def compute_point_count(self):
        """
        Computes the point count for all occupied voxels.

        This method calculates the number of points within each voxel and adds a new column 'point_count' to `self.df`.
        """
        if self.voxelized is False:
            self.voxelize()
            
        grouped = self.df.groupby(["voxel_x", "voxel_y", "voxel_z"])
        self.df["point_count"] = grouped["X"].transform("size")
        self.point_count = True

    @trace
    @timeit
    def compute_point_density(self):
        """
        Computes the point density for all occupied voxels.

        Adds a new column 'point_density' to `self.df`, calculated as point count divided by voxel volume.
        """
        if self.point_count is False:
            self.compute_point_count()
            if "point_count" not in self.compute:
                self.drop_columns += ["point_count"]
        self.df["point_density"] = self.df["point_count"] / (self.voxel_size**3)
        self.point_density = True

    @trace
    @timeit
    def compute_percentage_occupied(self):
        """
        Computes the percentage of space occupied by voxels within the voxel space bounding box.

        The percentage occupied is calculated as:
            Percentage occupied = (Number of occupied voxels / Total number of voxels in bounding box) * 100

        Notes
        -----
        - Requires the big integer index to be computed.
        - Prints the percentage of voxel space occupied.
        """
        if self.voxel_index is False:
            self.compute_voxel_index()
            # if "voxel_index" not in self.compute:
            #     self.drop_columns += ["voxel_index"]
        x_min, x_max = self.df["voxel_x"].min(), self.df["voxel_x"].max()
        y_min, y_max = self.df["voxel_y"].min(), self.df["voxel_y"].max()
        z_min, z_max = self.df["voxel_z"].min(), self.df["voxel_z"].max()
        x_extent, y_extent, z_extent = (
            x_max - x_min + 1,
            y_max - y_min + 1,
            z_max - z_min + 1,
        )
        nr_of_voxels_within_bounding_box = x_extent * y_extent * z_extent
        nr_of_occupied_voxels = len(self.df.index.unique()) #len(np.unique(self.df["voxel_index"]))
        self.percentage_occupied = round(
            nr_of_occupied_voxels / nr_of_voxels_within_bounding_box * 100, 2
        )
        # print(f"{self.percentage_occupied} percent of the voxel space is occupied")

    def compute_distance_to_center_of_gravity(self):
        """
        Computes the Euclidean distance from each point to the center of gravity of its voxel.

        This method calculates the distance between each point ('X', 'Y', 'Z') and the corresponding
        voxel's center of gravity ('cog_x', 'cog_y', 'cog_z').

        Adds a new column 'distance' to `self.df`.
        """
        if self.center_of_gravity is False:
            self.compute_center_of_gravity()
            if "center_of_gravity" not in self.compute:
                self.drop_columns += ["cog_x", "cog_y", "cog_z"]

        self.df["distance"] = np.sqrt(
            (self.df["X"] - self.df["cog_x"]) ** 2
            + (self.df["Y"] - self.df["cog_y"]) ** 2
            + (self.df["Z"] - self.df["cog_z"]) ** 2
        )
        self.distance_to_center_of_gravity = True

    @trace
    @timeit
    def compute_closest_to_center_of_gravity(self):
        """
        Identifies the point closest to the center of gravity within each voxel.

        This method determines which point in each voxel is nearest to the voxel's center of gravity
        and retains only those points.

        Notes
        -----
        - Adds 'min_distance' to `self.df`.
        - Merges the minimum distance information back into `self.df`.
        """
        if not self.center_of_gravity:
            self.compute_center_of_gravity()
            if "center_of_gravity" not in self.compute:
                self.drop_columns += ["cog_x", "cog_y", "cog_z"]
        if not self.distance_to_center_of_gravity:
            self.compute_distance_to_center_of_gravity()
            if "distance_to_center_of_gravity" not in self.compute:
                self.drop_columns += ["distance"]
        grouped = self.df.groupby(["cog_x", "cog_y", "cog_z"])
        self.voxel_cls2cog = grouped[["distance"]].min().reset_index()
        self.voxel_cls2cog.rename(columns={"distance": "min_distance"}, inplace=True)
        self.df = self.df.merge(
            self.voxel_cls2cog, how="left", on=["cog_x", "cog_y", "cog_z"]
        )
        self.closest_to_center_of_gravity = True

    @trace
    @timeit
    def compute_center_of_gravity(self):
        """
        Computes the center of gravity for all occupied voxels.

        Notes
        -----
        - Adds 'cog_x', 'cog_y', 'cog_z' to `self.df`.
        """
        if not self.voxelized:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]

        # Use groupby with transform to compute the mean of the X, Y, Z columns for each voxel,
        # and broadcast the computed means back to self.df without an merge.
        cog = self.df.groupby(["voxel_x", "voxel_y", "voxel_z"], sort=False
                              )[["X", "Y", "Z"]].transform("mean")
        
        # Rename the columns to the desired names.
        cog.columns = ["cog_x", "cog_y", "cog_z"]

        # Assign the computed center-of-gravity columns directly to the DataFrame.
        self.df[["cog_x", "cog_y", "cog_z"]] = cog

        self.center_of_gravity = True

    @trace
    @timeit
    def compute_std_of_cog(self):
        """
        Computes the standard deviation of the center of gravity for all occupied voxels.

        Notes
        -----
        - Adds 'std_x', 'std_y', 'std_z' to `self.df`.
        """
        # Ensure the DataFrame is voxelized; if not, voxelize and note columns to drop.
        if not self.voxelized:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]

        # --------------------------------------------------------------------------
        # CHANGES:
        # - Instead of performing a groupby followed by computing std, resetting index,
        #   renaming columns, and then merging back, we use groupby with transform.
        # - The transform method computes the standard deviation for each group and 
        #   broadcasts the results back to the original DataFrame's shape.
        # - This avoids the expensive merge operation.
        # --------------------------------------------------------------------------
        std_values = self.df.groupby(["voxel_x", "voxel_y", "voxel_z"], sort=False
                                     )[["X", "Y", "Z"]].transform("std")
        
        # Rename the resulting columns to reflect standard deviation values.
        std_values.columns = ["std_x", "std_y", "std_z"]

        # Directly assign the computed standard deviation columns to self.df.
        self.df[["std_x", "std_y", "std_z"]] = std_values

        # Mark that the standard deviation of the center of gravity has been computed.
        self.std_of_cog = True

    @trace
    @timeit
    def compute_center_of_voxel(self):
        """
        Computes the voxel center for all occupied voxels.

        Notes
        -----
        - Adds 'center_x', 'center_y', 'center_z' to `self.df`.
        """
        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
        self.df[["center_x", "center_y", "center_z"]] = (
            (self.df[["voxel_x", "voxel_y", "voxel_z"]] * self.voxel_size)
            + self.voxel_size / 2
            + self.origin
        )
        self.center_of_voxel = True

    @trace
    @timeit
    def compute_corner_of_voxel(self):
        """
        Computes the minx, miny, minz corner for all occupied voxels.

        Notes
        -----
        - Adds 'corner_x', 'corner_y', 'corner_z' to `self.df`.
        """
        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
        self.df[["corner_x", "corner_y", "corner_z"]] = (
            self.df[["voxel_x", "voxel_y", "voxel_z"]] * self.voxel_size
        ) + self.origin
        self.corner_of_voxel = True
    
    @trace
    @timeit
    def compute_covariance_matrix(self):
        """
        Computes the covariance matrix for each voxel using groupby.transform,
        avoiding an extra merge step.

        For each voxel (grouped by voxel_x, voxel_y, voxel_z), the following aggregates are computed:
            - n: number of points
            - sum_x, sum_y, sum_z: sums of coordinates
            - sum_x2, sum_y2, sum_z2: sums of squares
            - sum_xy, sum_xz, sum_yz: sums of cross products

        Then, using the formula:
            cov(x, y) = (sum_xy - (sum_x * sum_y)/n) / (n - 1)
        the covariance components are computed.
        """
        if not self.voxelized:
            self.voxelize()

        df = self.df

        # Precompute extra columns
        df["X2"] = df["X"] ** 2
        df["Y2"] = df["Y"] ** 2
        df["Z2"] = df["Z"] ** 2
        df["XY"] = df["X"] * df["Y"]
        df["XZ"] = df["X"] * df["Z"]
        df["YZ"] = df["Y"] * df["Z"]

        # Define the grouping keys
        group_keys = ["voxel_x", "voxel_y", "voxel_z"]
        grouped = df.groupby(group_keys)

        # Compute aggregates using transform so the result aligns with the original DataFrame.
        df["n"] = grouped["X"].transform("size")
        df["sum_x"] = grouped["X"].transform("sum")
        df["sum_y"] = grouped["Y"].transform("sum")
        df["sum_z"] = grouped["Z"].transform("sum")
        df["sum_x2"] = grouped["X2"].transform("sum")
        df["sum_y2"] = grouped["Y2"].transform("sum")
        df["sum_z2"] = grouped["Z2"].transform("sum")
        df["sum_xy"] = grouped["XY"].transform("sum")
        df["sum_xz"] = grouped["XZ"].transform("sum")
        df["sum_yz"] = grouped["YZ"].transform("sum")

        n = df["n"]

        # Compute covariance components.
        # For groups with n == 1, we set the covariance to NaN.
        df["cov_xx"] = np.where(
            n > 1,
            (df["sum_x2"] - (df["sum_x"] ** 2) / n) / (n - 1),
            np.nan
        )
        df["cov_yy"] = np.where(
            n > 1,
            (df["sum_y2"] - (df["sum_y"] ** 2) / n) / (n - 1),
            np.nan
        )
        df["cov_zz"] = np.where(
            n > 1,
            (df["sum_z2"] - (df["sum_z"] ** 2) / n) / (n - 1),
            np.nan
        )
        df["cov_xy"] = np.where(
            n > 1,
            (df["sum_xy"] - (df["sum_x"] * df["sum_y"]) / n) / (n - 1),
            np.nan
        )
        df["cov_xz"] = np.where(
            n > 1,
            (df["sum_xz"] - (df["sum_x"] * df["sum_z"]) / n) / (n - 1),
            np.nan
        )
        df["cov_yz"] = np.where(
            n > 1,
            (df["sum_yz"] - (df["sum_y"] * df["sum_z"]) / n) / (n - 1),
            np.nan
        )

        # Since the covariance matrix is symmetric, we assign the mirrored values:
        df["cov_yx"] = df["cov_xy"]
        df["cov_zx"] = df["cov_xz"]
        df["cov_zy"] = df["cov_yz"]

        # Optionally, drop temporary columns if you no longer need them.
        df.drop(
            columns=[
                "X2", "Y2", "Z2", "XY", "XZ", "YZ",
                "n", "sum_x", "sum_y", "sum_z",
                "sum_x2", "sum_y2", "sum_z2",
                "sum_xy", "sum_xz", "sum_yz"
            ],
            inplace=True
        )

        col_names = [
                    "cov_xx",
                    "cov_xy",
                    "cov_xz",
                    "cov_yx",
                    "cov_yy",
                    "cov_yz",
                    "cov_zx",
                    "cov_zy",
                    "cov_zz",
                ]
        # Reorder the DataFrame
        other_cols = [col for col in df.columns if col not in col_names]
        new_order = col_names + other_cols
        df = df[new_order]  
        self.df = df
        self.covariance_matrix_computed = True
        
    @trace
    @timeit
    def compute_eigenvalues(self):            #TODO: Change compute_eigenvalues to compute_eigenvalues_and_vectors in a similar way to the new implementation of compute_covariance_matrix
        """
        Computes eigenvalues for all occupied voxels.

        !!! Eigenvectors are also calculated, might be interesting to add to output

        Notes
        -----
        - Adds 'Eigenvalue_1', 'Eigenvalue_2', 'Eigenvalue_3' to `self.df`.
        """

        def _eigenvalues(df):
            cov_matrix = df[["X", "Y", "Z"]].cov()
            if cov_matrix.isna().any().any():
                eigenValues = np.array([np.nan, np.nan, np.nan])
            else:
                eigenValues, eigenVectors = np.linalg.eig(cov_matrix)
                # Currently sort eigenvalues but not the vectors

                idx = eigenValues.argsort()[::-1]
                eigenValues = eigenValues[idx]
            return eigenValues.flatten()

        if self.covariance_matrix is False:
            self.compute_covariance_matrix()
            if "covariance_matrix" not in self.compute:
                self.drop_columns += [
                    "cov_xx",
                    "cov_xy",
                    "cov_xz",
                    "cov_yx",
                    "cov_yy",
                    "cov_yz",
                    "cov_zx",
                    "cov_zy",
                    "cov_zz",
                ]
        grouped = self.df.groupby(["voxel_x", "voxel_y", "voxel_z"])
        eig_df = grouped.apply(_eigenvalues)
        col_names = ["Eigenvalue_1", "Eigenvalue_2", "Eigenvalue_3"]
        eigenvalue_df = pd.DataFrame(
            eig_df.values.tolist(), index=eig_df.index, columns=col_names
        ).reset_index()
        self.df = self.df.merge(
            eigenvalue_df, how="left", on=["voxel_x", "voxel_y", "voxel_z"]
        )
        self.eigenvalues = True

    @trace
    @timeit
    def compute_geometric_features(self):
        """
        Computes geometric features for all occupied voxels.
        http://dx.doi.org/10.1109/CVPR.2016.178
        Notes
        -----
        - Adds various geometric feature columns to `self.df`, such as 'Sum_of_Eigenvalues',
        'Omnivariance', 'Eigenentropy', 'Anisotropy', 'Planarity', 'Linearity',
        'Surface_Variation', and 'Sphericity'.
        """
        def safe_log(x):
            # Return 0 when x is 0; otherwise return the log.
            return np.where(x > 0, np.log(x), 0)
        if self.eigenvalues is False:
            self.compute_eigenvalues()
            if "eigenvalues" not in self.compute:
                self.drop_columns += ["Eigenvalue_1", "Eigenvalue_2", "Eigenvalue_3"]
        # Normalize eigenvalues so they sum to 1 for each row
        total = self.df["Eigenvalue_1"] + self.df["Eigenvalue_2"] + self.df["Eigenvalue_3"]
        self.df["Eigenvalue_1_n"] = self.df["Eigenvalue_1"] / total
        self.df["Eigenvalue_2_n"] = self.df["Eigenvalue_2"] / total
        self.df["Eigenvalue_3_n"] = self.df["Eigenvalue_3"] / total

        self.df["Sum_of_Eigenvalues"] = (
            self.df["Eigenvalue_1"] + self.df["Eigenvalue_2"] + self.df["Eigenvalue_3"]
        )
        self.df["Omnivariance"] = (
            self.df["Eigenvalue_1"] * self.df["Eigenvalue_2"] * self.df["Eigenvalue_3"]
        ) ** (1 / 3)
        try:
            # Now compute entropy using the normalized probabilities
            self.df["Eigenentropy"] = - (
                self.df["Eigenvalue_1_n"] * safe_log(self.df["Eigenvalue_1_n"])
                + self.df["Eigenvalue_2_n"] * safe_log(self.df["Eigenvalue_2_n"])
                + self.df["Eigenvalue_3_n"] * safe_log(self.df["Eigenvalue_3_n"])
            )
        except:
            self.df["Eigenentropy"] = np.nan
        self.df.drop(["Eigenvalue_1_n", "Eigenvalue_2_n","Eigenvalue_3_n"], axis=1)
        self.df["Anisotropy"] = (
            self.df["Eigenvalue_1"] - self.df["Eigenvalue_3"]
        ) / self.df["Eigenvalue_1"]
        self.df["Planarity"] = (
            self.df["Eigenvalue_2"] - self.df["Eigenvalue_3"]
        ) / self.df["Eigenvalue_1"]
        self.df["Linearity"] = (
            self.df["Eigenvalue_1"] - self.df["Eigenvalue_2"]
        ) / self.df["Eigenvalue_1"]
        self.df["Surface_Variation"] = (
            self.df["Eigenvalue_3"] / self.df["Sum_of_Eigenvalues"]
        )
        self.df["Sphericity"] = self.df["Eigenvalue_3"] / self.df["Eigenvalue_1"]
        self.geometric_features = True

    @trace
    @timeit
    def reduce_to_voxels(self):
        """
        Reduces the DataFrame to only one value per voxel. `return_at` defines what the X, Y, and Z coordinate
        of the output will be.
        `return_at` overwrites X, Y, Z with:
        - The center of each voxel containing points ("center_of_voxel")
        - The minx, miny, minz corner of each voxel containing points ("corner_of_voxel")
        - The center of gravity computed within each voxel containing points ("center_of_gravity")
        - The point closest to the center of gravity computed within each voxel containing points ("closest_to_center_of_gravity")

        Notes
        -----
        - Removes voxel columns and updates coordinate columns based on `return_at`.
        - Drops duplicate entries.
        - Sets `self.reduced` to True after reduction.
        """
        if self.return_at == "center_of_voxel":
            if self.center_of_voxel is False or not hasattr(self.df, "center_x"):
                self.compute_center_of_voxel()
                self.drop_columns += ["center_x", "center_y", "center_z"]
            self.new_column_names.update(
                {"X": "center_x", "Y": "center_y", "Z": "center_z"}
            )

        elif self.return_at == "corner_of_voxel":
            if self.corner_of_voxel is False:
                self.compute_corner_of_voxel()
                self.drop_columns += ["corner_x", "corner_y", "corner_z"]
            self.new_column_names.update(
                {"X": "corner_x", "Y": "corner_y", "Z": "corner_z"}
            )

        elif self.return_at == "center_of_gravity":
            if not self.center_of_gravity:
                self.compute_center_of_gravity()
                self.drop_columns += ["cog_x", "cog_y", "cog_z"]
            self.new_column_names.update({"X": "cog_x", "Y": "cog_y", "Z": "cog_z"})

        elif self.return_at == "closest_to_center_of_gravity":
            if not self.center_of_gravity:
                self.compute_center_of_gravity()
                self.drop_columns += ["cog_x", "cog_y", "cog_z"]
            if not self.closest_to_center_of_gravity:
                self.compute_closest_to_center_of_gravity()
                self.drop_columns += ["min_distance"]
            self.df = self.df[self.df["distance"] == self.df["min_distance"]]
        else:
            print(
                f"Voxels cannot be reduced to {self.return_at},\n \
                    try 'center_of_gravity', 'center_of_voxel', 'closest_to_center_of_gravity', or 'corner_of_voxel'"
            )
            return
        # Update columns with their required values
        for col_name in self.new_column_names.keys():
            self.df[col_name] = self.df[self.new_column_names[col_name]]

        self.df = self.df.drop(set(self.drop_columns), axis=1)

        self.df.drop_duplicates(subset=["X", "Y", "Z"], inplace=True)
        self.df = self.df.groupby(["X", "Y", "Z"], as_index=False).median(numeric_only=True)
        self.reduced = True

    @trace
    @timeit
    def filter_attributes(self, filter_attribute: str, min_max_eq: str, filter_value):
        """
        Filters a DataFrame attribute based on specified criteria.
        This method modifies the DataFrame `self.df` by applying a filter condition based on the specified attribute, value, and filter type.

        Choice of operators: ['equal_to', 'greater_than', 'less_than', 'greater_than_or_equal_to', 'less_than_or_equal_to', '==', '>', '<', '>=', '<=']

        Parameters:
        ----------
        filter_attribute : str
            The attribute (column name) of the DataFrame to apply the filter on.
        filter_value : int, float
            The value to compare the attribute against. Must be compatible with the type of the DataFrame attribute.
        min_max_eq : str 
            A string specifying the type of filter to apply.
        
        Returns
        -------
        None
        
        Raises
        ------
        ValueError
            If the `min_max_eq` parameter is not one of the valid filter strings.
        KeyError
            If the `filter_attribute` is not present in the DataFrame.

        Example:
        ```
        # Assuming `self.df` is a DataFrame with a column 'point_count'
        self.filter_attributes('point_count', 'greater_than_or_equal_to', 30)
        # This will modify `self.df` to include only rows where 'point_count' is 30 or more.
        ```
        """
        valid_strings = [
            "equal_to", "==",
            "not_equal_to", "!=",
            "greater_than", ">",
            "greater_than_or_equal_to", ">=",
            "less_than", "<",
            "less_than_or_equal_to", "<=",
        ]
        min_max_eq = min_max_eq.lower()
        if min_max_eq in ["equal_to", "=="]:
            self.df = self.df[self.df[filter_attribute] == filter_value]
        elif min_max_eq in ["not_equal_to", "!="]:
            self.df = self.df[self.df[filter_attribute] != filter_value]
        elif min_max_eq in ["greater_than", ">"]:
            self.df = self.df[self.df[filter_attribute] > filter_value]
        elif min_max_eq in ["greater_than_or_equal_to", ">="]:
            self.df = self.df[self.df[filter_attribute] >= filter_value]
        elif min_max_eq in ["less_than", "<"]:
            self.df = self.df[self.df[filter_attribute] < filter_value]
        elif min_max_eq in ["less_than_or_equal_to", "<="]:
            self.df = self.df[self.df[filter_attribute] <= filter_value]
        else:
            raise ValueError(f"Filter invalid, use only one of the following:\n\n{', '.join(valid_strings)}.")

    @trace
    @timeit
    def compute_clusters(
        self, cluster_distance, cluster_by=None
    ):
        """
        Groups data points into clusters based on spatial proximity within a specified distance.

        This method identifies clusters in the point cloud data where points are within
        `cluster_distance` of each other. It utilizes the KDTree algorithm for efficient
        neighbor searches. Each cluster is assigned a unique ID, and the size of each
        cluster is calculated.

        Parameters
        ----------
        cluster_distance : float
            The maximum distance between points to be considered part of the same cluster.
        cluster_by : list of str, optional
            List of column names in `self.df` that represent spatial coordinates used
            for clustering. Defaults to ["voxel_x", "voxel_y", "voxel_z"].

        Returns
        -------
        None

        Updates
        -------
        self.df : pandas.DataFrame
            The DataFrame is updated in-place and includes two new columns:
            - "cluster_id": An integer representing the cluster ID for each point.
            - "cluster_size": The number of points in the cluster to which each point belongs.

        Notes
        -----
        - The method uses a fixed number of nearest neighbors (`nr_of_neighbours = 50`)
        during the KDTree query.
        - Clusters are merged if they share points within the specified `cluster_distance`.
        - An internal function `_update_existing_objects` is defined to handle the
        merging of overlapping clusters.

        Raises
        ------
        KeyError
            If any of the columns specified in `cluster_by` are not present in `self.df`.

        Examples
        --------
        >>> vapc_instance.compute_clusters(cluster_distance=1.0)
        >>> print(vapc_instance.df[['cluster_id', 'cluster_size']])
        """
        if cluster_by is None:
            cluster_by = ["voxel_x", "voxel_y", "voxel_z"]
        if not self.voxelized:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
        def _update_existing_objects(oc, indices, relevantPoints):
            obj, counts = np.unique(oc[indices[relevantPoints]], return_counts=True)
            existingObjects = obj[obj > 0]
            for existingObject in existingObjects:
                mask = np.where(oc == existingObject)
                oc[mask] = existingObjects.min()
            oc[indices[relevantPoints]] = existingObjects.min()

        nr_of_neighbours = 50
        pts = np.array(self.df[cluster_by])
        tree = KDTree(pts)
        oc = np.zeros_like(self.df[cluster_by[0]])
        self.objectCounter = 1.0

        for i, point in enumerate(pts):
            distances, indices = tree.query(point, nr_of_neighbours)
            relevantPoints = np.where(distances <= cluster_distance)
            if oc[indices[relevantPoints]].max() < 1:
                oc[indices[relevantPoints]] = self.objectCounter
                self.objectCounter += 1
            else:
                _update_existing_objects(oc, indices, relevantPoints)

        oids, cts = np.unique(oc, return_counts=True)
        ct_df = pd.DataFrame(
            data=np.array((oids, cts)).T, columns=["cluster_id", "cluster_size"]
        )
        self.df["cluster_id"] = oc
        self.df = self.df.merge(
            ct_df, on="cluster_id", how="left", validate="many_to_one"
        )
