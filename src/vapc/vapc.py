# For Vapc:
import numpy as np
import math
from scipy.spatial import KDTree
import pandas as pd
from .utilities import trace, timeit


class Vapc:
    AVAILABLE_COMPUTATIONS = [
        "big_int_index",
        "hash_index",
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
            "big_int_index",
            "hash_index",
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

        # Calculations not applied yet:
        self.attributes_up_to_data = False
        self.voxelized = False
        self.big_int_index = False
        self.hash_index = False
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
        if not hasattr(data_handler, "df"):
            raise AttributeError(
                "The provided data_handler does not have a 'df' attribute."
            )
        self.df = data_handler.df
        del data_handler.df  # Remove df from data_handler
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
            - "big_int_index"
            - "hash_index"
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
        # import time

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
                    elif stat == "median":
                        value = np.median(group_slice)
                    elif stat == "min":
                        value = group_slice.min()
                    elif stat == "max":
                        value = group_slice.max()
                    elif stat == "sum":
                        value = group_slice.sum()
                    elif stat == "mode":
                        counts = np.bincount(group_slice.astype(int))
                        value = np.argmax(counts)
                    elif "mode_count" in stat:
                        # Extract percentage threshold
                        try:
                            _, percentage_str = stat.split(",")
                            percentage = float(percentage_str)
                        except ValueError:
                            raise ValueError(
                                f"Invalid 'mode_count' specification for attribute '{attr}': '{stat}'"
                            )
                        counts = np.bincount(group_slice.astype(int))
                        counts_sum = counts.sum()
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
    def compute_big_int_index(self, n=1000000000):
        """
        Computes a big int index for all occupied voxels (as a int).
        Do not set n > 1000000000, errors will occur.

        Parameters
        ----------
        n : int, optional
            The base multiplier for voxel indexing. Defaults to 1000000000.

        Notes
        -----
        - Adds a new column 'big_int_index' to `self.df`.
        """
        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]

        self.df.loc[:, "big_int_index"] = (
            self.df.loc[:, "voxel_x"] * n**2
            + self.df.loc[:, "voxel_y"] * n
            + self.df.loc[:, "voxel_z"]
        )
        self.big_int_index = True

    @trace
    @timeit
    def compute_voxel_index(self):
        """
        Computes a unique voxel index using voxel coordinates as a tuple.

        Notes
        -----
        - Adds a new column 'voxel_index' to `self.df`.
        """
        if not self.voxelized:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]

        self.df["voxel_index"] = list(
            zip(self.df["voxel_x"], self.df["voxel_y"], self.df["voxel_z"])
        )
        self.voxel_index = True

    @trace
    @timeit
    def compute_voxel_buffer(self, buffer_size: int = 1):
        """
        Computes a buffer around each voxel by expanding voxel coordinates within a specified buffer size.

        This method generates new voxel coordinates that are within the buffer distance from the original voxels.
        The buffer includes all neighboring voxels within the given buffer size in all three dimensions.

        Parameters
        ----------
        buffer_size : int, optional
            The size of the buffer around each voxel. Defaults to 1.

        Updates
        -------
        self.buffer_df : pandas.DataFrame
            A DataFrame containing the expanded voxel coordinates with columns ['voxel_x', 'voxel_y', 'voxel_z'].

        Notes
        -----
        - The original DataFrame `self.df` remains unchanged.
        - This method sets `self.voxelized` to True if voxelization is performed.
        """

        # Ensure the data is voxelized
        if not self.voxelized:
            self.voxelize()

        required_columns = ["voxel_x", "voxel_y", "voxel_z"]
        coords = (
            self.df[required_columns].drop_duplicates().values
        )  # Avoid duplicate voxels

        # Generate offset combinations
        offsets = np.arange(-buffer_size, buffer_size + 1)
        all_combinations = np.array(np.meshgrid(offsets, offsets, offsets)).T.reshape(
            -1, 3
        )

        # Expand coordinates by adding offsets
        expanded_coords = (
            coords[:, np.newaxis, :] + all_combinations
        )  # Shape: (num_voxels, num_offsets, 3)
        result = expanded_coords.reshape(
            -1, 3
        )  # Flatten to (num_voxels * num_offsets, 3)

        # Remove duplicate coordinates
        result_df = pd.DataFrame(result, columns=required_columns).drop_duplicates()

        # Optionally, remove coordinates outside the original data bounds if necessary

        # Store the buffer coordinates in a new attribute
        self.buffer_df = result_df.reset_index(drop=True)

    @trace
    @timeit
    def select_by_mask(
        self, vapc_mask, mask_attribute="voxel_index", segment_in_or_out="in"
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

        if not hasattr(self, "df") or not hasattr(vapc_mask, "df"):
            raise AttributeError("Both `self.df` and `vapc_mask.df` must exist.")

        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]

        if vapc_mask.voxelized is False:
            vapc_mask.voxelize()

        if mask_attribute not in self.df.columns:
            # Attempt to compute the attribute
            if hasattr(self, f"compute_{mask_attribute}"):
                getattr(self, f"compute_{mask_attribute}")()
            else:
                raise AttributeError(
                    f"Attribute '{mask_attribute}' not found and cannot be computed."
                )

        if mask_attribute not in vapc_mask.df.columns:
            if hasattr(vapc_mask, f"compute_{mask_attribute}"):
                getattr(vapc_mask, f"compute_{mask_attribute}")()
            else:
                raise AttributeError(
                    f"Attribute '{mask_attribute}' not found in `vapc_mask` and cannot be computed."
                )

        # mask by attribute
        if segment_in_or_out == "in":
            mask_values = set(vapc_mask.df[mask_attribute])
            self.df = self.df[self.df[mask_attribute].isin(mask_values)].reset_index(
                drop=True
            )
        elif segment_in_or_out == "out":
            mask_values = set(vapc_mask.df[mask_attribute])
            self.df = self.df[~self.df[mask_attribute].isin(mask_values)].reset_index(
                drop=True
            )
        else:
            raise ValueError(
                "Parameter 'segment_in_or_out' must be either 'in' or 'out'."
            )

        # print("Points after filtering:",self.df.shape)
        self.df = self.df.drop(
            ["voxel_x", "voxel_y", "voxel_z", mask_attribute], axis=1
        )
        self.voxelized = False

    @trace
    @timeit
    def compute_hash_index(
        self, p1=76690892503, p2=15752609759, p3=27174879103, n=2**100
    ):
        """
        Computes the hash index for all occupied voxels.

        Parameters
        ----------
        p1, p2, p3 : int, optional
            Large prime numbers used in the hashing function.
        n : int, optional
            Modulus for the hashing function. Defaults to 2**100.

        Notes
        -----
        - This method adds a new column 'hash_index' to `self.df`.
        - Be cautious with the size of `n` and the prime numbers to avoid integer overflows.
        """
        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
        self.df["hash_index"] = (
            (self.df["voxel_x"] * p1)
            ^ (self.df["voxel_y"] * p2)
            ^ (self.df["voxel_z"] * p3)
        ) % n
        self.hash_index = True

    @trace
    @timeit
    def compute_point_count(self):
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
    def compute_point_density(self):
        """
        Computes the point density for all occupied voxels.

        Adds a new column 'point_density' to `self.df`, calculated as point count divided by voxel volume.
        """
        if self.point_count is False:
            self.compute_point_count()
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
            self.drop_columns += ["voxel_index"]
        x_min, x_max = self.df["voxel_x"].min(), self.df["voxel_x"].max()
        y_min, y_max = self.df["voxel_y"].min(), self.df["voxel_y"].max()
        z_min, z_max = self.df["voxel_z"].min(), self.df["voxel_z"].max()
        x_extent, y_extent, z_extent = (
            x_max - x_min + 1,
            y_max - y_min + 1,
            z_max - z_min + 1,
        )
        nr_of_voxels_within_bounding_box = x_extent * y_extent * z_extent
        nr_of_occupied_voxels = len(np.unique(self.df["voxel_index"]))
        self.percentage_occupied = round(
            nr_of_occupied_voxels / nr_of_voxels_within_bounding_box * 100, 2
        )
        print("%s percent of the voxel space is occupied" % self.percentage_occupied)

    def compute_distance_to_center_of_gravity(self):
        """
        Computes the Euclidean distance from each point to the center of gravity of its voxel.

        This method calculates the distance between each point ('X', 'Y', 'Z') and the corresponding
        voxel's center of gravity ('cog_x', 'cog_y', 'cog_z').

        Adds a new column 'distance' to `self.df`.
        """
        if self.center_of_gravity is False:
            self.compute_center_of_gravity()
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
            self.drop_columns += ["cog_x", "cog_y", "cog_z"]
        if not self.distance_to_center_of_gravity:
            self.compute_distance_to_center_of_gravity()
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
        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
        grouped = self.df.groupby(["voxel_x", "voxel_y", "voxel_z"])
        self.voxel_cog = grouped[["X", "Y", "Z"]].mean().reset_index()
        self.voxel_cog.rename(
            columns={"X": "cog_x", "Y": "cog_y", "Z": "cog_z"}, inplace=True
        )
        self.df = self.df.merge(
            self.voxel_cog, how="left", on=["voxel_x", "voxel_y", "voxel_z"]
        )
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
        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
        grouped = self.df.groupby(["voxel_x", "voxel_y", "voxel_z"])
        self.voxel_cog = grouped[["X", "Y", "Z"]].std().reset_index()
        self.voxel_cog.rename(
            columns={"X": "std_x", "Y": "std_y", "Z": "std_z"}, inplace=True
        )
        self.df = self.df.merge(
            self.voxel_cog, how="left", on=["voxel_x", "voxel_y", "voxel_z"]
        )
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
        Computes the covariance matrix for all occupied voxels.

        Notes
        -----
        - Adds covariance matrix components ('cov_xx', 'cov_xy', ..., 'cov_zz') to `self.df`.
        """

        def _covariance(df):
            cov_matrix = df[["X", "Y", "Z"]].cov()
            return cov_matrix.values.flatten()

        if self.voxelized is False:
            self.voxelize()
            self.drop_columns += ["voxel_x", "voxel_y", "voxel_z"]
        grouped = self.df.groupby(["voxel_x", "voxel_y", "voxel_z"])
        cov_df = grouped.apply(_covariance)
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
        covariance_df = pd.DataFrame(
            cov_df.values.tolist(), index=cov_df.index, columns=col_names
        ).reset_index()
        self.df = self.df.merge(
            covariance_df, how="left", on=["voxel_x", "voxel_y", "voxel_z"]
        )
        self.covariance_matrix = True

    @trace
    @timeit
    def compute_eigenvalues(self):
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

        Notes
        -----
        - Adds various geometric feature columns to `self.df`, such as 'Sum_of_Eigenvalues',
        'Omnivariance', 'Eigentropy', 'Anisotropy', 'Planarity', 'Linearity',
        'Surface_Variation', and 'Sphericity'.
        """
        if self.eigenvalues is False:
            self.compute_eigenvalues()
            self.drop_columns += ["Eigenvalue_1", "Eigenvalue_2", "Eigenvalue_3"]
        self.df["Sum_of_Eigenvalues"] = (
            self.df["Eigenvalue_1"] + self.df["Eigenvalue_2"] + self.df["Eigenvalue_3"]
        )
        self.df["Omnivariance"] = (
            self.df["Eigenvalue_1"] * self.df["Eigenvalue_2"] * self.df["Eigenvalue_3"]
        ) ** (1 / 3)
        try:
            self.df["Eigentropy"] = -1 * (
                self.df["Eigenvalue_1"] * math.log(self.df["Eigenvalue_1"])
                + self.df["Eigenvalue_2"] * math.log(self.df["Eigenvalue_2"])
                + self.df["Eigenvalue_3"] * math.log(self.df["Eigenvalue_3"])
            )
        except:
            self.df["Eigentropy"] = np.nan
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
                f"Voxels cannot be reduced to {self.return_at},try 'center_of_gravity', 'center_of_voxel', 'closest_to_center_of_gravity', or 'corner_of_voxel'"
            )
            return

        # Update columns with their required values
        for col_name in self.new_column_names.keys():
            self.df[col_name] = self.df[self.new_column_names[col_name]]
        self.df = self.df.drop(set(self.drop_columns), axis=1)
        self.df = self.df.drop_duplicates()
        self.df = self.df.groupby(["X", "Y", "Z"], as_index=False).median()
        self.reduced = True

    @trace
    @timeit
    def filter_attributes(
        self,
        filter_attribute: str,
        min_max_eq: str,
        filter_value,
    ):
        """
        Filters a DataFrame attribute based on specified criteria.
        This method modifies the DataFrame `self.df` by applying a filter condition based on the specified attribute, value, and filter type.
        Filters include:
            equality ('eq')
            greater than ('min')
            greater than or equal ('min_eq')
            less than ('max')
            less than or equal ('max_eq')

        Parameters:
        - filter_attribute (str): The attribute (column name) of the DataFrame to apply the filter on.
        - filter_value (Comparable): The value to compare the attribute against. Must be compatible with the type of the DataFrame attribute.
        - min_max_eq (str): A string specifying the type of filter to apply.

        Example:
        ```
        # Assuming `self.df` is a DataFrame with a column 'point_count'
        self.filter_attributes('point_count', 'min_eq', 30)
        # This will modify `self.df` to include only rows where 'point_count' is 30 or more.
        ```
        """
        min_max_eq = min_max_eq.lower()
        if min_max_eq == "eq":
            self.df = self.df[self.df[filter_attribute] == filter_value]
        elif min_max_eq == "min":
            self.df = self.df[self.df[filter_attribute] > filter_value]
        elif min_max_eq == "min_eq":
            self.df = self.df[self.df[filter_attribute] >= filter_value]
        elif min_max_eq == "max":
            self.df = self.df[self.df[filter_attribute] < filter_value]
        elif min_max_eq == "max_eq":
            self.df = self.df[self.df[filter_attribute] <= filter_value]
        else:
            print("Filter invalid, use eq, min, min_eq, max, and max_eq only.")

    @trace
    @timeit
    def compute_clusters(
        self, cluster_distance, cluster_by=["voxel_x", "voxel_y", "voxel_z"]
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

        def _update_existing_objects(obj, indices, relevantPoints):
            obj, counts = np.unique(obj[indices[relevantPoints]], return_counts=True)
            existingObjects = obj[obj > 0]
            for existingObject in existingObjects:
                mask = np.where(obj == existingObject)
                obj[mask] = existingObjects.min()
            obj[indices[relevantPoints]] = existingObjects.min()

        nr_of_neighbours = 50
        pts = np.array(self.df[cluster_by])
        tree = KDTree(pts)
        obj = np.zeros_like(self.df[cluster_by[0]])
        self.objectCounter = 1.0

        for i, point in enumerate(pts):
            distances, indices = tree.query(point, nr_of_neighbours)
            relevantPoints = np.where(distances <= cluster_distance)
            if obj[indices[relevantPoints]].max() < 1:
                obj[indices[relevantPoints]] = self.objectCounter
                self.objectCounter += 1
            else:
                _update_existing_objects(obj, indices, relevantPoints)

        oids, cts = np.unique(obj, return_counts=True)
        ct_df = pd.DataFrame(
            data=np.array((oids, cts)).T, columns=["cluster_id", "cluster_size"]
        )
        self.df["cluster_id"] = obj
        self.df = self.df.merge(
            ct_df, on="cluster_id", how="left", validate="many_to_one"
        )
