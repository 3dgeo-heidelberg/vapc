import pytest
import vapc
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs

RED_POINT = [120, 200, 3]

@pytest.fixture
def input_df_only_geom():
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
def vapc_dataset_clustering():
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
    # initiate vapc object
    vapc_obj = vapc.Vapc(
        voxel_size=1
    )
    # set input dataframe
    vapc_obj.df = input_df_only_geom
    vapc_obj.original_attributes = vapc_obj.df.columns.tolist()

    return vapc_obj


def test_compute_reduction_point(vapc_dataset_geom_50cm):
    vapc_dataset_geom_50cm.compute_reduction_point()
    assert vapc_dataset_geom_50cm.reduction_point == RED_POINT


def test_compute_offset(vapc_dataset_geom_50cm):
    assert vapc_dataset_geom_50cm.offset_applied == False
    vapc_dataset_geom_50cm.compute_offset()
    assert vapc_dataset_geom_50cm.offset_applied == True
    assert vapc_dataset_geom_50cm.df["X"].min() == 0
    assert vapc_dataset_geom_50cm.df["Y"].min() == 0
    assert vapc_dataset_geom_50cm.df["Z"].min() == 0


def test_compute_offset_reverse(vapc_dataset_geom_50cm):
    vapc_dataset_geom_50cm.compute_offset()
    assert vapc_dataset_geom_50cm.offset_applied is True
    vapc_dataset_geom_50cm.compute_offset()
    assert vapc_dataset_geom_50cm.offset_applied is False
    assert vapc_dataset_geom_50cm.df["X"].min() == RED_POINT[0]
    assert vapc_dataset_geom_50cm.df["Y"].min() == RED_POINT[1]
    assert vapc_dataset_geom_50cm.df["Z"].min() == RED_POINT[2]


def test_compute_clusters(vapc_dataset_clustering):
    vapc_dataset_clustering.compute_clusters(cluster_distance=10.0)
    assert "cluster_id" in vapc_dataset_clustering.df.columns
    assert "cluster_size" in vapc_dataset_clustering.df.columns
    assert vapc_dataset_clustering.df["cluster_id"].nunique() == 3


def test_filter_attributes_invalid_filter(vapc_dataset_geom_1m):
    vapc_dataset_geom_1m.voxelize()
    vapc_dataset_geom_1m.compute = ["point_count"]
    vapc_dataset_geom_1m.compute_requested_attributes()
    with pytest.raises(ValueError):
        vapc_dataset_geom_1m.filter_attributes(filter_attribute="point_count", min_max_eq="leq", filter_value=3)


def test_filter_attributes_invalid_attribute(vapc_dataset_geom_1m):
    with pytest.raises(KeyError):
        vapc_dataset_geom_1m.filter_attributes(filter_attribute="attribute", min_max_eq="<=", filter_value=3)


@pytest.mark.parametrize("filter_attribute,filter_value,operator,expected_num_points", 
                         [
                             ["point_count", 3, ">", 442],
                         ])
def test_filter_attributes(vapc_dataset_geom_1m, filter_attribute, filter_value, operator, expected_num_points):
    vapc_dataset_geom_1m.voxelize()
    vapc_dataset_geom_1m.compute = [filter_attribute]
    vapc_dataset_geom_1m.compute_requested_attributes()
    vapc_dataset_geom_1m.filter_attributes(filter_attribute=filter_attribute, min_max_eq=operator, filter_value=filter_value)
    assert vapc_dataset_geom_1m.df.shape[0] == expected_num_points
