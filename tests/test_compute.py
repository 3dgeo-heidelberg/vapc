import pytest
from unittest.mock import patch, call
import vapc
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs


RED_POINT = [120, 200, 3]


def get_expected_columns(attributes):
    """
    Get the expected columns after computing requested attributes.
    """
    if not isinstance(attributes, list):
        attributes = [attributes]
    if "center_of_gravity" in attributes:
        attributes.remove("center_of_gravity")
        attributes = ["cog_x", "cog_y", "cog_z"]
    if "distance_to_center_of_gravity" in attributes:
        attributes.remove("distance_to_center_of_gravity")
        attributes += ["distance"]
    if "closest_to_center_of_gravity" in attributes:
        attributes.remove("closest_to_center_of_gravity")
        attributes += ["min_distance"]
    if "covariance_matrix" in attributes:
        attributes.remove("covariance_matrix")
        attributes += [
            "cov_xx",
            "cov_xy",
            "cov_xz",
            "cov_yx",
            "cov_yy",
            "cov_yz",
            "cov_zx",
            "cov_zy",
            "cov_zz"
            ]
    if "eigenvalues" in attributes:
        attributes.remove("eigenvalues")
        attributes += ["Eigenvalue_1", "Eigenvalue_2", "Eigenvalue_3"]
    if "geometric_features" in attributes:
        attributes.remove("geometric_features")
        attributes += [
            "Sum_of_Eigenvalues",
            "Omnivariance",
            "Eigentropy",
            "Anisotropy",
            "Planarity",
            "Linearity",
            "Surface_Variation",
            "Sphericity"
        ]
    if "std_of_cog" in attributes:
        attributes.remove("std_of_cog")
        attributes += ["std_x", "std_y", "std_z"]
    if "center_of_voxel" in attributes:
        attributes.remove("center_of_voxel")
        attributes += ["center_x", "center_y", "center_z"]
    if "corner_of_voxel" in attributes:
        attributes.remove("corner_of_voxel")
        attributes += ["corner_x", "corner_y", "corner_z"]
    if "percentage_occupied" in attributes:
        attributes = [None]
    return attributes


@pytest.fixture
def input_df_only_geom():
    """
    Fixture for a dataframe with only geometric attributes (x, y, z coordinates).
    """
    # create artificial dataframe for testing
    # starting with 500 points on a plane
    np.random.seed(42)
    x_coords = np.random.uniform(0.0, 10.0, 500)
    y_coords = np.random.uniform(0.0, 10.0, 500)
    z_coords = np.zeros(500)
    # add noise to z coords
    z_coords += np.random.normal(0.0, 0.1, 500)
    # create dataframe
    df = pd.DataFrame({
        "X": x_coords,
        "Y": y_coords,
        "Z": z_coords
    })
    # set minimum coords to RED_POINT
    df["X"] -= df["X"].min() - RED_POINT[0]
    df["Y"] -= df["Y"].min() - RED_POINT[1]
    df["Z"] -= df["Z"].min() - RED_POINT[2]
    
    return df


@pytest.fixture
def input_df_with_intensity_nor():
    """
    Fixture for a dataframe with geometry (x, y, z), intensity and number of returns attributes.
    """
    # create artificial dataframe for testing
    # starting with 500 points on a plane
    np.random.seed(42)
    x_coords = np.random.uniform(0.0, 10.0, 500)
    y_coords = np.random.uniform(0.0, 10.0, 500)
    z_coords = np.zeros(500)
    # add noise to z coords
    z_coords += np.random.normal(0.0, 0.1, 500)
    # simulated intensity (normal distribution, mean=-5, std=5)
    intensity = np.random.normal(-3, 5, 500)
    # simulated NOR (random int between 1 and 5)
    nor = np.random.randint(1, 5, 500)

    # create dataframe
    df = pd.DataFrame({
        "X": x_coords,
        "Y": y_coords,
        "Z": z_coords,
        "intensity": intensity,
        "number_of_returns": nor
    })
    # set minimum coords to RED_POINT
    df["X"] -= df["X"].min() - RED_POINT[0]
    df["Y"] -= df["Y"].min() - RED_POINT[1]
    df["Z"] -= df["Z"].min() - RED_POINT[2]
    
    return df


@pytest.fixture
def vapc_dataset_clustering():
    """
    Fixture for a synthetic vapc object for testing clustering.
    """
    # Parameters for the synthetic dataset
    n_samples = 300    # Total number of samples
    n_features = 3     # Number of dimensions (3D points)
    centers = 3        # Number of clusters
    cluster_std = 1.0  # Standard deviation of clusters

    # Generate the synthetic dataset
    pts, cluster_id = make_blobs(n_samples=n_samples,
                                 n_features=n_features,
                                 centers=centers,
                                 cluster_std=cluster_std,
                                 random_state=42)

    # create dataframe
    df = pd.DataFrame({
        "X": pts[:, 0],
        "Y": pts[:, 1],
        "Z": pts[:, 2],
        "cluster_id": cluster_id
    })

    # initiate vapc object
    vapc_obj = vapc.Vapc(
        voxel_size=0.2
    )
    # set input dataframe
    vapc_obj.df = df
    vapc_obj.original_attributes = vapc_obj.df.columns.tolist()

    return vapc_obj


@pytest.fixture
def vapc_dataset_geom_50cm(input_df_only_geom):
    """
    Fixture for a synthetic vapc object with only geometry (x, y, z)
    and a voxel size of 0.5 m.
    """
    # initiate vapc object
    vapc_obj = vapc.Vapc(
        voxel_size=0.5
    )
    # set input dataframe
    vapc_obj.df = input_df_only_geom
    vapc_obj.original_attributes = vapc_obj.df.columns.tolist()

    return vapc_obj


@pytest.fixture
def vapc_dataset_geom_1m(input_df_only_geom):
    """
    Fixture for a synthetic vapc object with only geometry (x, y, z)
    and a voxel size of 1 m.
    """

    # initiate vapc object
    vapc_obj = vapc.Vapc(
        voxel_size=1
    )
    # set input dataframe
    vapc_obj.df = input_df_only_geom
    vapc_obj.original_attributes = vapc_obj.df.columns.tolist()

    return vapc_obj


def test_compute_reduction_point(vapc_dataset_geom_50cm):
    """
    Test the computation of the reduction point.
    """
    vapc_dataset_geom_50cm.compute_reduction_point()
    assert vapc_dataset_geom_50cm.reduction_point == RED_POINT


def test_compute_offset(vapc_dataset_geom_50cm):
    """
    Test the computation of the offset.
    """
    assert vapc_dataset_geom_50cm.offset_applied is False
    vapc_dataset_geom_50cm.compute_offset()
    assert vapc_dataset_geom_50cm.offset_applied is True
    assert vapc_dataset_geom_50cm.df["X"].min() == 0
    assert vapc_dataset_geom_50cm.df["Y"].min() == 0
    assert vapc_dataset_geom_50cm.df["Z"].min() == 0


def test_compute_offset_reverse(vapc_dataset_geom_50cm):
    """
    Test the computation of the offset and its reversal.
    """
    vapc_dataset_geom_50cm.compute_offset()
    assert vapc_dataset_geom_50cm.offset_applied is True
    vapc_dataset_geom_50cm.compute_offset()
    assert vapc_dataset_geom_50cm.offset_applied is False
    assert vapc_dataset_geom_50cm.df["X"].min() == RED_POINT[0]
    assert vapc_dataset_geom_50cm.df["Y"].min() == RED_POINT[1]
    assert vapc_dataset_geom_50cm.df["Z"].min() == RED_POINT[2]


def test_compute_clusters(vapc_dataset_clustering):
    """
    Test the computation of clusters.
    """
    vapc_dataset_clustering.compute_clusters(cluster_distance=10.0)
    assert "cluster_id" in vapc_dataset_clustering.df.columns
    assert "cluster_size" in vapc_dataset_clustering.df.columns
    assert vapc_dataset_clustering.df["cluster_id"].nunique() == 3


# TODO: check if belongs here or rather integration test
@pytest.mark.parametrize("attribute",
                         ["big_int_index",
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
                          "corner_of_voxel"]
)
def test_compute_attributes(vapc_dataset_geom_1m, attribute):
    """
    Test the computation of requested attributes by mocking the
    requested method and making sure it is called once.
    """
    vapc_dataset_geom_1m.voxelize()
    vapc_dataset_geom_1m.compute = [attribute]
    method_name = f"vapc.Vapc.compute_{attribute}"
    with patch(method_name) as mock_method:
        vapc_dataset_geom_1m.compute_requested_attributes()
        mock_method.assert_called_once()


@pytest.mark.parametrize(
    "attribute,stats",
    [
        ["intensity", ["mean", "min", "max", "median", "sum", "mode"]],
        ["number_of_returns", ["mode", "mode_count,0.1"]]
    ]
)
def test_compute_requested_statistics_per_attributes(input_df_with_intensity_nor, attribute, stats):
    """
    Test the computation of requested statistics per attributes.
    """
    # initiate vapc object
    vapc_obj = vapc.Vapc(
        voxel_size=2.5
    )
    # set input dataframe and attributes
    vapc_obj.df = input_df_with_intensity_nor
    vapc_obj.original_attributes = input_df_with_intensity_nor.columns.tolist()
    vapc_obj.attributes = {attribute: stats}
    # compute voxel_index and requested statistics
    vapc_obj.compute_voxel_index()
    vapc_obj.compute_requested_statistics_per_attributes()
    for stat in stats:
        # check that the new columns are added
        assert f"{attribute}_{stat}" in vapc_obj.df.columns
        # check that points in the same voxel have the same value
        assert vapc_obj.df.groupby("voxel_index")[f"{attribute}_{stat}"].nunique().max() == 1
        # check that the computed statistics are correct
        if stat == "mean":
            assert np.isclose(vapc_obj.df[f"{attribute}_{stat}"].mean(), input_df_with_intensity_nor[attribute].mean(), atol=1e-08, equal_nan=False)
        elif stat == "min":
            assert np.isclose(vapc_obj.df[f"{attribute}_{stat}"].min(), input_df_with_intensity_nor[attribute].min(), atol=1e-08, equal_nan=False)
        elif stat == "max":
            assert np.isclose(vapc_obj.df[f"{attribute}_{stat}"].max(), input_df_with_intensity_nor[attribute].max(), atol=1e-08, equal_nan=False)


def test_filter_attributes_invalid_filter(vapc_dataset_geom_1m):
    """
    Test that a Value Error is raised when an invalid 
    filter string (`min_max_eq`) is provided.
    """
    vapc_dataset_geom_1m.voxelize()
    vapc_dataset_geom_1m.compute = ["point_count"]
    vapc_dataset_geom_1m.compute_requested_attributes()
    with pytest.raises(ValueError):
        vapc_dataset_geom_1m.filter_attributes(filter_attribute="point_count",
                                               min_max_eq="leq",
                                               filter_value=3)


def test_filter_attributes_invalid_attribute(vapc_dataset_geom_1m):
    """
    Test that a Key Error is raised when an invalid filter attribute is provided.
    """
    with pytest.raises(KeyError):
        vapc_dataset_geom_1m.filter_attributes(filter_attribute="attribute",
                                               min_max_eq="<=",
                                               filter_value=3)


@pytest.mark.parametrize("filter_attribute,filter_value,operator,expected_num_points", 
                         [
                             ["point_count", 3, ">", 442],
                         ])
def test_filter_attributes(vapc_dataset_geom_1m, filter_attribute, filter_value, operator, expected_num_points):
    """
    Test the filtering of attributes.
    """
    vapc_dataset_geom_1m.voxelize()
    # set and compute attribute
    vapc_dataset_geom_1m.compute = [filter_attribute]
    vapc_dataset_geom_1m.compute_requested_attributes()
    # filter
    vapc_dataset_geom_1m.filter_attributes(filter_attribute=filter_attribute, min_max_eq=operator, filter_value=filter_value)
    # check number of points
    assert vapc_dataset_geom_1m.df.shape[0] == expected_num_points
