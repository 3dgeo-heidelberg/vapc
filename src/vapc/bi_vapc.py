import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.stats import chi2

from .utilities import trace, timeit
from .datahandler import DataHandler

@trace
@timeit
class BiTemporalVapc:
    def __init__(self, vapcs: list):
        if len(vapcs) != 2:
            raise ValueError("BiTemporalVapc requires exactly two dataframes.")
        self.vapcs = vapcs
        self.merged_df = None
        self.final_df = None
        self.distance_computed = False
        self.mahalanobis_computed = False
        self.prepared_for_bi_temporal_analysis = False
        self.single_occupation_computed = False
    
    @trace
    @timeit
    def compute_voxels_occupied_in_single_epoch(self):
        # Identify unique voxels for each epoch.
        # Return error stating data should be prepared for bi_temporal analysis.
        if self.prepared_for_bi_temporal_analysis == False:
            raise ValueError("Data should be prepared for bi_temporal analysis using 'prepare_data_for_mahalanobis_distance()'")
        voxels_occupied_only_in_epoch_1 = self.vapcs[0].df.loc[self.vapcs[0].df.index.difference(self.vapcs[1].df.index)]
        voxels_occupied_only_in_epoch_2 = self.vapcs[1].df.loc[self.vapcs[1].df.index.difference(self.vapcs[0].df.index)]
        voxels_occupied_only_in_epoch_1["change_type"] = 3 #disappearing
        voxels_occupied_only_in_epoch_2["change_type"] = 4 #appearing
        # Merge occupied in single epoch only
        self.df_appear_disappear = pd.concat([voxels_occupied_only_in_epoch_1, voxels_occupied_only_in_epoch_2])
        self.single_occupation_computed = True

    @trace
    @timeit
    def prepare_data_for_mahalanobis_distance(self):
        # Map original coordinates to center-of-gravity columns for both epochs.
        for vapc in self.vapcs:
            vapc.return_at = "center_of_gravity"
            vapc.compute_center_of_gravity()
            vapc.compute_covariance_matrix()
            vapc.reduce_to_voxels()
        self.prepared_for_bi_temporal_analysis = True

    @trace
    @timeit
    def merge_vapcs_with_closest_point(self):
        df1, df2 = self.vapcs[0].df, self.vapcs[1].df

        # Get indices of mutual nearest neighbors.
        idx1, idx2 = mutual_nearest_neighbors(df1, df2)

        # Reduce the DataFrames to mutual points.
        df1_reduced = df1.iloc[idx1].copy()
        df2_reduced = df2.iloc[idx2].copy()

        # Recalculate distances between the mutual pairs.
        tree2_reduced = KDTree(df2_reduced[["cog_x", "cog_y", "cog_z"]])
        distances_reduced, _ = tree2_reduced.query(df1_reduced[["cog_x", "cog_y", "cog_z"]], k=1)
        df1_reduced["distance"] = distances_reduced

        # Reset indices before merging.
        df1_reduced.reset_index(drop=True, inplace=True)
        df2_reduced.reset_index(drop=True, inplace=True)

        # Merge using the helper.
        self.merged_df = merge_with_suffix(df1_reduced, df2_reduced)
        # Optionally rename the "distance" column.
        self.merged_df.rename(columns={"distance_x": "distance"}, inplace=True)
        self.distance_computed = True
    
    @trace
    @timeit
    def merge_vapcs_with_same_voxel_index(self):
        for vapc in self.vapcs:
            vapc.voxelize()
            vapc.compute_voxel_index()
        df1, df2 = self.vapcs[0].df, self.vapcs[1].df
        # Merge based on index and add suffixes.
        self.merged_df = df1.add_suffix("_x").join(df2.add_suffix("_y"), how='inner')
        
    def compute_distance(self):
        self.merged_df["distance"] = compute_euclidean_distance(self.merged_df)
        self.distance_computed = True

    @trace
    @timeit
    def compute_mahalanobis_distance(self, alpha: float = 0.005):
        if self.merged_df is None:
            raise ValueError("Run compute_nearest_neighbors() before computing Mahalanobis test.")

        df = self.merged_df
        x1 = df[["cog_x_x", "cog_y_x", "cog_z_x"]].to_numpy()
        x2 = df[["cog_x_y", "cog_y_y", "cog_z_y"]].to_numpy()

        # Get covariance matrix columns.
        cov_cols_x = sorted([col for col in df.columns if col.startswith("cov_") and col.endswith("_x")])
        cov_cols_y = sorted([col for col in df.columns if col.startswith("cov_") and col.endswith("_y")])
        if len(cov_cols_x) != 9 or len(cov_cols_y) != 9:
            raise ValueError("Incomplete covariance matrix columns.")

        cov_x = df[cov_cols_x].to_numpy().reshape(-1, 3, 3)
        cov_y = df[cov_cols_y].to_numpy().reshape(-1, 3, 3)
        eps = 1e-10  # Numerical stability
        cov_x += np.eye(3) * eps
        cov_y += np.eye(3) * eps

        inv_cov_x = np.linalg.inv(cov_x)
        inv_cov_y = np.linalg.inv(cov_y)
        diff = x1 - x2

        d1 = np.einsum("ij,ijk,ik->i", diff, inv_cov_y, diff)
        d2 = np.einsum("ij,ijk,ik->i", diff, inv_cov_x, diff)
        p_val1 = 1 - chi2.cdf(d1, df=3)
        p_val2 = 1 - chi2.cdf(d2, df=3)

        is_outlier = (p_val1 < alpha) | (p_val2 < alpha)
        p_value = np.where(p_val1 < alpha, p_val1, p_val2)

        df["mahalanobi_significance"] = is_outlier.astype(int)
        df["p_value"] = p_value
        
        # Points with p_value == 0 have less than 30 points in either one or both epochs. As we do
        # not compute the mahalanobis distance for these points, we set the change_type to 3 so that
        # these areas are considered as significant. This is a conservative approach but in line with 
        # the handling of appearing and disappearing areas.

        df["change_type"] = 1 #change detected
        df.loc[df["mahalanobi_significance"] == 0, "change_type"] = 0 #no change detected
        # df.loc[df["mahalanobi_significance"] == 1, "change_type"] = 1 #change detected
        df.loc[df["p_value"] == 0, "change_type"] = 2 #less than 30 points in one or both epochs

        self.merged_df = df
        self.mahalanobis_computed = True
    
    @trace
    @timeit
    def prepare_data_for_export(self):
        if self.merged_df is None:
            raise ValueError("Run compute_mahalanobis_distance() before saving to LAS.")
        #create dataframe with X,Y,Z, and further columns, that have been computed
        #in the bi-temporal VAPC
        relevant_columns = ["cog_x_x", "cog_y_x", "cog_z_x"]
        if self.distance_computed:
            relevant_columns.append("distance")
        if self.mahalanobis_computed:
            relevant_columns.extend(["mahalanobi_significance", "p_value","change_type"])
        df = self.merged_df[relevant_columns]
        #rename columns from cog_x_x, cog_y_X to ...X, ...Y
        self.df = df.rename(columns={
                "cog_x_x": "X",
                "cog_y_x": "Y",
                "cog_z_x": "Z"
            })

        if self.single_occupation_computed == True:
            self.df = pd.concat([self.df, self.df_appear_disappear[["X","Y","Z","change_type"]]])

    @trace
    @timeit
    def save_to_las(self, output_file: str):
        """Save the DataFrame to a LAS file."""
        dh = DataHandler("")
        dh.df = self.df
        dh.save_as_las(output_file)

@trace
@timeit
def mutual_nearest_neighbors(df1, df2, coord_cols=["cog_x", "cog_y", "cog_z"]):
    """Find mutual nearest neighbor indices between two DataFrames."""
    coords1 = df1[coord_cols].values
    coords2 = df2[coord_cols].values

    # Forward search: from df1 to df2.
    tree2 = KDTree(coords2)
    _, indices = tree2.query(coords1, k=1)

    # Reverse check: from df2 (selected via indices) back to df1.
    tree1 = KDTree(coords1)
    _, back_indices = tree1.query(coords2[indices], k=1)
    mutual_mask = np.arange(len(df1)) == back_indices
    return np.arange(len(df1))[mutual_mask], indices[mutual_mask]

@trace
@timeit
def merge_with_suffix(df1, df2, suffix1="_x", suffix2="_y"):
    """Merge two DataFrames side-by-side with column suffixes."""
    df1_renamed = df1.add_suffix(suffix1)
    df2_renamed = df2.add_suffix(suffix2)
    return pd.concat([df1_renamed, df2_renamed], axis=1)

@trace
@timeit
def compute_euclidean_distance(df, coord_prefix="cog"):
    """Compute Euclidean distance between two sets of coordinates in a merged DataFrame.
    
    Assumes columns like 'cog_x_x', 'cog_y_x', 'cog_z_x' (from first DataFrame)
    and 'cog_x_y', 'cog_y_y', 'cog_z_y' (from second DataFrame).
    """
    dx = df[f"{coord_prefix}_x_x"] - df[f"{coord_prefix}_x_y"]
    dy = df[f"{coord_prefix}_y_x"] - df[f"{coord_prefix}_y_y"]
    dz = df[f"{coord_prefix}_z_x"] - df[f"{coord_prefix}_z_y"]
    return np.sqrt(dx**2 + dy**2 + dz**2)