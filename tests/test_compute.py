import pytest
import vapc
import pandas as pd
import numpy as np

RED_POINT = [120, 200, 3]

@pytest.fixture
def input_df_only_geom():
    # create artificial dataframe for testing
    # starting with 500 points on a plane
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
def vapc_dataset_geom_20cm(input_df_only_geom):
    # initiate vapc object
    vapc_obj = vapc.Vapc(
        voxel_size=0.2
    )
    # set input dataframe
    vapc_obj.df = input_df_only_geom
    vapc_obj.original_attributes = vapc_obj.df.columns.tolist()

    return vapc_obj


def test_compute_reduction_point(vapc_dataset_geom_20cm):
    vapc_dataset_geom_20cm.compute_reduction_point()
    assert vapc_dataset_geom_20cm.reduction_point == RED_POINT


def test_compute_offset(vapc_dataset_geom_20cm):
    assert vapc_dataset_geom_20cm.offset_applied == False
    vapc_dataset_geom_20cm.compute_offset()
    assert vapc_dataset_geom_20cm.offset_applied == True
    assert vapc_dataset_geom_20cm.df["X"].min() == 0
    assert vapc_dataset_geom_20cm.df["Y"].min() == 0
    assert vapc_dataset_geom_20cm.df["Z"].min() == 0


def test_compute_offset_reverse(vapc_dataset_geom_20cm):
    vapc_dataset_geom_20cm.compute_offset()
    assert vapc_dataset_geom_20cm.offset_applied == True
    vapc_dataset_geom_20cm.compute_offset()
    assert vapc_dataset_geom_20cm.offset_applied == False
    assert vapc_dataset_geom_20cm.df["X"].min() == RED_POINT[0]
    assert vapc_dataset_geom_20cm.df["Y"].min() == RED_POINT[1]
    assert vapc_dataset_geom_20cm.df["Z"].min() == RED_POINT[2]
