import pytest
import vapc
import pandas as pd
import numpy as np


VOXEL_X = [0, 1, 1, 3, 4, 5, 6, 9]
VOXEL_Y = [0, 1, 2, 3, 5, 2, 5, 7]
VOXEL_Z = [0, 1, 0, 0, 1, 0, 2, 1]
VOXEL_SIZE = 1


@pytest.fixture
def input_df_from_voxel():
    # known voxel grid, 8 voxels

    # create coordinates: 10 points per voxel randomly distributed
    x_coords = []
    y_coords = []
    z_coords = []
    for vx, vy, vz in zip(VOXEL_X, VOXEL_Y, VOXEL_Z):
        x_coords += list(np.random.uniform(vx, vx + VOXEL_SIZE, 5))
        y_coords += list(np.random.uniform(vy, vy + VOXEL_SIZE, 5))
        z_coords += list(np.random.uniform(vz, vz + VOXEL_SIZE, 5))
    
    # create dataframe
    df = pd.DataFrame({
        "X": x_coords,
        "Y": y_coords,
        "Z": z_coords
    })

    return df


@pytest.fixture
def vapc_dataset_from_voxel(input_df_from_voxel):
    # initiate vapc object
    vapc_obj = vapc.Vapc(
        voxel_size=VOXEL_SIZE
    )
    # set input dataframe
    vapc_obj.df = input_df_from_voxel
    vapc_obj.original_attributes = vapc_obj.df.columns.tolist()

    return vapc_obj


def test_voxelize(vapc_dataset_from_voxel):
    vapc_dataset_from_voxel.voxelize()
    assert vapc_dataset_from_voxel.voxelized is True
    found_voxels = np.empty((vapc_dataset_from_voxel.df.shape[0], 3))
    for v_idx in range(vapc_dataset_from_voxel.df.shape[0]):
        voxel = vapc_dataset_from_voxel.df.iloc[v_idx]
        found_voxels[v_idx, :] = (voxel.voxel_x, voxel.voxel_y, voxel.voxel_z)
    found_voxels = np.unique(found_voxels, axis=0)
    assert found_voxels.shape[0] == len(VOXEL_X)
    np.testing.assert_equal(found_voxels, list(zip(VOXEL_X, VOXEL_Y, VOXEL_Z)))
