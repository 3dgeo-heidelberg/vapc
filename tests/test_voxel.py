import pytest
import vapc
import pandas as pd
import numpy as np
import scipy
from itertools import product


VOXEL_X = [0, 1, 1, 3, 4, 5, 6, 9]
VOXEL_Y = [0, 1, 2, 3, 5, 2, 5, 7]
VOXEL_Z = [0, 1, 0, 0, 1, 0, 2, 1]
VOXEL_SIZE = 1
PPTS_PER_VOXEL = 5
MASK_X = [0, 0, 1, 1]
MASK_Y = [0, 1, 1, 2]
MASK_Z = [0, 0, 1, 0]


@pytest.fixture()
def input_df_from_voxel():
    # known voxel grid, 8 voxels

    # create coordinates: 5 points per voxel randomly distributed
    x_coords = []
    y_coords = []
    z_coords = []
    for vx, vy, vz in zip(VOXEL_X, VOXEL_Y, VOXEL_Z):
        x_coords += list(np.random.uniform(vx, vx + VOXEL_SIZE, PPTS_PER_VOXEL))
        y_coords += list(np.random.uniform(vy, vy + VOXEL_SIZE, PPTS_PER_VOXEL))
        z_coords += list(np.random.uniform(vz, vz + VOXEL_SIZE, PPTS_PER_VOXEL))
    
    # create dataframe
    df = pd.DataFrame({
        "X": x_coords,
        "Y": y_coords,
        "Z": z_coords
    })

    return df


@pytest.fixture()
def mask_df_from_voxel():
    # known voxel grid, 4 voxels

    # create coordinates: 5 points per voxel randomly distributed
    x_coords = []
    y_coords = []
    z_coords = []
    for vx, vy, vz in zip(MASK_X, MASK_Y, MASK_Z):
        x_coords += list(np.random.uniform(vx, vx + VOXEL_SIZE, PPTS_PER_VOXEL))
        y_coords += list(np.random.uniform(vy, vy + VOXEL_SIZE, PPTS_PER_VOXEL))
        z_coords += list(np.random.uniform(vz, vz + VOXEL_SIZE, PPTS_PER_VOXEL))
    
    # create dataframe
    df = pd.DataFrame({
        "X": x_coords,
        "Y": y_coords,
        "Z": z_coords
    })

    return df


@pytest.fixture()
def vapc_dataset_factory():
    def _create(input_df):
        # initiate vapc object
        vapc_obj = vapc.Vapc(
            voxel_size=VOXEL_SIZE
        )
        # set input dataframe
        vapc_obj.df = input_df
        vapc_obj.original_attributes = vapc_obj.df.columns.tolist()
        return vapc_obj
    return _create


def test_voxelize(vapc_dataset_factory, input_df_from_voxel):
    # voxelize dataset
    vapc_dataset = vapc_dataset_factory(input_df_from_voxel)
    vapc_dataset.voxelize()
    assert vapc_dataset.voxelized is True
    # expected voxels
    voxels = list(zip(VOXEL_X, VOXEL_Y, VOXEL_Z))
    # actual voxels
    found_voxels = np.empty((vapc_dataset.df.shape[0], 3))
    for v_idx in range(vapc_dataset.df.shape[0]):
        voxel = vapc_dataset.df.iloc[v_idx]
        found_voxels[v_idx, :] = (voxel.voxel_x, voxel.voxel_y, voxel.voxel_z)
    found_voxels = np.unique(found_voxels, axis=0)
    # assertions
    assert found_voxels.shape[0] == len(VOXEL_X)
    np.testing.assert_equal(found_voxels, voxels)


def test_compute_voxel_index(vapc_dataset_factory, input_df_from_voxel):
    vapc_dataset = vapc_dataset_factory(input_df_from_voxel)
    vapc_dataset.compute_voxel_index()
    # expected indices
    unique_voxels = np.column_stack([VOXEL_X, VOXEL_Y, VOXEL_Z])
    voxel_idx = np.repeat(unique_voxels, PPTS_PER_VOXEL, axis=0)
    # actual indices
    actual_idx = vapc_dataset.df[["voxel_index"]].values
    actual_idx = np.array(actual_idx)
    # reshape
    actual_idx = np.array([list(row[0]) for row in actual_idx])
    # assertions
    assert vapc_dataset.voxel_index is True
    assert voxel_idx.shape[0] == actual_idx.shape[0]
    np.testing.assert_equal(actual_idx, voxel_idx)


@pytest.mark.parametrize("n", [100000, 1000000000])
def test_compute_big_int_index(vapc_dataset_factory, input_df_from_voxel, n):
    vapc_dataset = vapc_dataset_factory(input_df_from_voxel)
    vapc_dataset.compute_big_int_index(n=n)
    assert vapc_dataset.big_int_index is True
    # expected indices
    unique_voxels = np.column_stack([VOXEL_X, VOXEL_Y, VOXEL_Z])
    big_int_idx = (
        unique_voxels[:, 0] * n**2
        + unique_voxels[:, 1] * n
        + unique_voxels[:, 2]
        )
    big_int_idx = np.repeat(big_int_idx, PPTS_PER_VOXEL)
    # actual indices
    actual_idx = vapc_dataset.df["big_int_index"].values
    assert big_int_idx.shape[0] == actual_idx.shape[0]
    np.testing.assert_equal(actual_idx, big_int_idx)


@pytest.mark.parametrize("n", [100000, 1000000000])
def test_compute_hash_index(vapc_dataset_factory, input_df_from_voxel, n):
    p1 = 76690892503
    p2 = 15752609759
    p3 = 27174879103
    vapc_dataset = vapc_dataset_factory(input_df_from_voxel)
    vapc_dataset.compute_hash_index(p1=p1, p2=p2, p3=p3, n=n)
    assert vapc_dataset.hash_index is True
    # expected indices
    unique_voxels = np.column_stack([VOXEL_X, VOXEL_Y, VOXEL_Z])
    hash_idx = (
        (unique_voxels[:, 0] * p1)
        ^ (unique_voxels[:, 1] * p2)
        ^ (unique_voxels[:, 2] * p3)
        ) % n
    hash_idx = np.repeat(hash_idx, PPTS_PER_VOXEL)
    # actual indices
    actual_idx = vapc_dataset.df["hash_index"].values
    assert hash_idx.shape[0] == actual_idx.shape[0]
    np.testing.assert_equal(actual_idx, hash_idx)


def test_compute_voxel_corner(vapc_dataset_factory, input_df_from_voxel):
    vapc_dataset = vapc_dataset_factory(input_df_from_voxel)
    # expected corners
    unique_corners = np.column_stack([VOXEL_X, VOXEL_Y, VOXEL_Z])
    corners = np.repeat(unique_corners, PPTS_PER_VOXEL, axis=0)
    # actual corners
    vapc_dataset.compute_corner_of_voxel()
    actual_corners = vapc_dataset.df[["corner_x", "corner_y", "corner_z"]].values
    # assertions
    assert vapc_dataset.corner_of_voxel is True
    assert actual_corners.shape[0] == len(corners)
    np.testing.assert_equal(actual_corners, corners)


def test_compute_voxel_buffer(vapc_dataset_factory, input_df_from_voxel):
    buffer_size = 1
    vapc_dataset = vapc_dataset_factory(input_df_from_voxel)
    vapc_dataset.compute_voxel_buffer(buffer_size=buffer_size)
    # expected buffer
    voxel_indices = np.column_stack([VOXEL_X, VOXEL_Y, VOXEL_Z])
    # Generate relative offsets for the buffer
    offsets = np.array(list(product(range(-buffer_size, buffer_size + 1), repeat=3)))
    # Expand each voxel index by the offsets and reshape
    buffered_voxels = np.array(voxel_indices[:, None] + offsets).reshape(-1, 3)
    buffered_voxels = np.unique(buffered_voxels, axis=0)
    # actual buffer
    buffer = np.unique(vapc_dataset.buffer_df[["voxel_x", "voxel_y", "voxel_z"]].values, axis=0)
    # assertions
    np.testing.assert_equal(buffer, buffered_voxels)
    assert buffer.shape[0] == buffered_voxels.shape[0]


def test_select_by_mask_in(vapc_dataset_factory, input_df_from_voxel, mask_df_from_voxel):
    # mask and voxelize the dataset
    vapc_dataset = vapc_dataset_factory(input_df_from_voxel)
    mask_dataset = vapc_dataset_factory(mask_df_from_voxel)
    vapc_dataset.select_by_mask(mask_dataset, segment_in_or_out="in")
    vapc_dataset.voxelize()
    # expected voxels
    voxels = [[x, y, z] for x, y, z in zip(MASK_X, MASK_Y, MASK_Z) if (x, y, z) in zip(VOXEL_X, VOXEL_Y, VOXEL_Z)]
    # actual voxels
    found_voxels = np.empty((vapc_dataset.df.shape[0], 3))
    for v_idx in range(vapc_dataset.df.shape[0]):
        voxel = vapc_dataset.df.iloc[v_idx]
        found_voxels[v_idx, :] = (voxel.voxel_x, voxel.voxel_y, voxel.voxel_z)
    found_voxels = np.unique(found_voxels, axis=0)
    # assertions
    # number of points
    assert vapc_dataset.df.shape[0] == len(voxels) * PPTS_PER_VOXEL
    # number of voxels
    assert found_voxels.shape[0] == len(voxels)
    # matching voxels
    np.testing.assert_equal(found_voxels, voxels)


def test_select_by_mask_out(vapc_dataset_factory, input_df_from_voxel, mask_df_from_voxel):
    # mask and voxelize the dataset
    vapc_dataset = vapc_dataset_factory(input_df_from_voxel)
    mask_dataset = vapc_dataset_factory(mask_df_from_voxel)
    vapc_dataset.select_by_mask(mask_dataset, segment_in_or_out="out")
    vapc_dataset.voxelize()
    # expected voxels
    voxels = [[x, y, z] for x, y, z in zip(VOXEL_X, VOXEL_Y, VOXEL_Z) if (x, y, z) not in zip(MASK_X, MASK_Y, MASK_Z)]
    # actual voxels
    found_voxels = np.empty((vapc_dataset.df.shape[0], 3))
    for v_idx in range(vapc_dataset.df.shape[0]):
        voxel = vapc_dataset.df.iloc[v_idx]
        found_voxels[v_idx, :] = (voxel.voxel_x, voxel.voxel_y, voxel.voxel_z)
    found_voxels = np.unique(found_voxels, axis=0)
    # assertions
    # number of points
    assert vapc_dataset.df.shape[0] == len(voxels) * PPTS_PER_VOXEL
    # number of voxels
    assert found_voxels.shape[0] == len(voxels)
    # matching voxels
    np.testing.assert_equal(found_voxels, voxels)


def test_subsampling_voxel_center(vapc_dataset_factory, input_df_from_voxel):
    vapc_dataset = vapc_dataset_factory(input_df_from_voxel)
    vapc_dataset.return_at = "center_of_voxel"
    vapc_dataset.reduce_to_voxels()
    # expected points
    points = np.array(list(zip(VOXEL_X, VOXEL_Y, VOXEL_Z))) + VOXEL_SIZE / 2
    # actual points
    reduced_points = vapc_dataset.df[["X", "Y", "Z"]].values
    assert vapc_dataset.df.shape[0] == len(points)
    np.testing.assert_equal(reduced_points, points)


def test_subsampling_cog(vapc_dataset_factory, input_df_from_voxel):
    vapc_dataset = vapc_dataset_factory(input_df_from_voxel)
    vapc_dataset.return_at = "center_of_gravity"
    vapc_dataset.reduce_to_voxels()
    # expected points
    averages = input_df_from_voxel.rolling(PPTS_PER_VOXEL).mean()[4::PPTS_PER_VOXEL]
    points = averages[["X", "Y", "Z"]].values
    points = points[np.lexsort(points.T[::-1])]
    # actual points
    reduced_points = vapc_dataset.df[["X", "Y", "Z"]].values
    reduced_points = reduced_points[np.lexsort(reduced_points.T[::-1])]
    # assertions
    assert vapc_dataset.df.shape[0] == len(points)
    np.testing.assert_allclose(reduced_points, points)


def test_subsampling_closest_to_cog(vapc_dataset_factory, input_df_from_voxel):
    vapc_dataset = vapc_dataset_factory(input_df_from_voxel)
    vapc_dataset.return_at = "closest_to_center_of_gravity"
    vapc_dataset.reduce_to_voxels()
    # expected points
    averages = input_df_from_voxel.rolling(PPTS_PER_VOXEL).mean()[4::PPTS_PER_VOXEL]
    points = averages[["X", "Y", "Z"]].values
    # find closest points
    dist = scipy.spatial.distance.cdist(points, input_df_from_voxel[["X", "Y", "Z"]].values)
    min_idx = np.argmin(dist, axis=1)
    points = input_df_from_voxel.iloc[min_idx][["X", "Y", "Z"]].values
    points = points[np.lexsort(points.T[::-1])]
    # actual points
    reduced_points = vapc_dataset.df[["X", "Y", "Z"]].values
    reduced_points = reduced_points[np.lexsort(reduced_points.T[::-1])]
    # assertions
    assert vapc_dataset.df.shape[0] == len(points)
    np.testing.assert_allclose(reduced_points, points)
