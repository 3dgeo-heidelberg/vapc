import pytest
import numpy as np
import vapc
from pathlib import Path
import laspy
from laspy.errors import LaspyException
import plyfile


@pytest.fixture
def test_file_path_1():
    return Path(__file__).parent.parent / "test_data" / "vapc_in.laz"


@pytest.fixture
def test_file_path_2():
    return Path(__file__).parent.parent / "test_data" / "small_mask.laz"


@pytest.fixture
def test_file_path_vls():
    return Path(__file__).parent.parent / "test_data" / "als_multichannel_leg000_points.laz"


@pytest.fixture
def datahandler_factory():
    def _create(input_file):
        dh = vapc.DataHandler(input_file)
        dh.load_las_files()
        return dh
    return _create


@pytest.fixture
def voxel_grid_laz_factory():
    def _create(input_file, tmp_path, voxel_size):
        dh = vapc.DataHandler(input_file)
        dh.load_las_files()
        vapc_dataset = vapc.Vapc(voxel_size=voxel_size,
                                 return_at="center_of_voxel")
        vapc_dataset.get_data_from_data_handler(dh)
        vapc_dataset.voxelize()
        output_file = tmp_path / "voxel_grid.laz"
        vapc_dataset.reduce_to_voxels()
        dh.df = vapc_dataset.df
        dh.save_as_las(output_file)
        return output_file
    return _create


@pytest.fixture
def temp_text_file(tmp_path):
    """
    Pytest fixture to create a temporary text file.
    Returns the file path.
    """
    file_path = tmp_path / "example.txt"
    file_content = "x y z intensity\n1.45 7.386 2.957 1300.423\n"
    file_path.write_text(file_content)
    return file_path


@pytest.mark.parametrize("to_string", [False, True])
def test_load_las_single_file(test_file_path_1, to_string):
    if to_string:
        test_file_path_1 = str(test_file_path_1)
    dh = vapc.DataHandler(test_file_path_1)
    # expected
    shape = (547662, 12)
    # assertions
    dh.load_las_files()
    assert dh.df is not None
    assert dh.df.shape == shape
    assert dh.attributes == ['intensity', 'bit_fields', 'raw_classification', 'scan_angle_rank', 'user_data', 'point_source_id', 'red', 'green', 'blue']
    assert dh.las_header is not None
    assert dh.df.columns.isin(['X', 'Y', 'Z']).any()


def test_load_las_vls_file(test_file_path_vls):
    dh = vapc.DataHandler(test_file_path_vls)
    # expected
    shape = (61283, 15)
    # assertions
    dh.load_las_files()
    assert dh.df is not None
    assert dh.df.shape == shape
    assert dh.attributes == ['intensity', 'bit_fields', 'classification_flags', 'classification', 'user_data', 'scan_angle', 'point_source_id', 'gps_time', 'echo_width', 'fullwaveIndex', 'hitObjectId', 'heliosAmplitude']
    assert dh.las_header is not None
    assert dh.df.columns.isin(['X', 'Y', 'Z']).any()


def test_load_las_multiple_files(test_file_path_1, test_file_path_2):
    dh = vapc.DataHandler([test_file_path_1, test_file_path_2])
    dh.load_las_files()
    # expected
    shape = (547662 + 151, 12)
    # assertions
    assert dh.df is not None
    assert dh.df.shape == shape
    assert dh.attributes == ['intensity', 'bit_fields', 'raw_classification', 'scan_angle_rank', 'user_data', 'point_source_id', 'red', 'green', 'blue']
    assert dh.las_header is not None
    assert dh.df.columns.isin(['X', 'Y', 'Z']).any()


def test_load_las_file_not_exists():
    dh = vapc.DataHandler("file_not_exists.laz")
    # TODO: Add extra error handling for invalid file extensions?
    with pytest.raises(FileNotFoundError):
        dh.load_las_files()


def test_load_las_file_one_not_exists(test_file_path_1):
    dh = vapc.DataHandler([test_file_path_1, "file_not_exists.laz"])
    # TODO: Add extra error handling for invalid file extensions?
    with pytest.raises(FileNotFoundError):
        dh.load_las_files()


@pytest.mark.parametrize("input", [300, [1, 2, 3], {"a": 1, "b": 2}, 3.14, None, True])
def test_load_invalid_data_type(input):
    dh = vapc.DataHandler(input)
    # TODO: Validate data type and return custom error?
    with pytest.raises(AttributeError):
        dh.load_las_files()


def test_load_file_invalid_file_type(temp_text_file):
    dh = vapc.DataHandler(temp_text_file)
    # TODO: Add extra error handling for invalid file extensions?
    with pytest.raises(LaspyException):
        dh.load_las_files()


@pytest.mark.parametrize("infile", ["test_file_path_1", "test_file_path_vls"])
def test_write_las_file(datahandler_factory, infile, request, tmp_path):
    output_file = tmp_path / "output.laz"
    infile = request.getfixturevalue(infile)
    dh = datahandler_factory(infile)
    dh.save_as_las(outfile=output_file,
                   las_point_format=7,
                   las_version="1.4")
    assert output_file.exists()
    # read with laspy
    las = laspy.read(infile)
    las_out = laspy.read(output_file)
    # assertions
    assert las_out.header.major_version == 1
    assert las_out.header.minor_version == 4
    assert las_out.header.point_format.id == 7
    assert len(las.points) == len(las_out.points)


@pytest.mark.parametrize("shift_to_voxel_center", [False, True])
def test_write_ply_file(tmp_path, voxel_grid_laz_factory, test_file_path_2, datahandler_factory, shift_to_voxel_center):
    output_file = tmp_path / "output.ply"
    voxel_size = 2
    infile = voxel_grid_laz_factory(test_file_path_2, tmp_path, voxel_size)
    dh = datahandler_factory(infile)
    dh.save_as_ply(outfile=output_file, voxel_size=voxel_size,
                   shift_to_center=shift_to_voxel_center)
    assert output_file.exists()
    # read with plyfile
    plydata = plyfile.PlyData.read(output_file)
    # assert vertex count
    assert plydata.elements[0].data.shape[0] == dh.df.shape[0] * 8
    # assert face count
    assert plydata.elements[1].data.shape[0] == dh.df.shape[0] * 6 * 2
    # assert properties
    properties = [p.name for p in plydata.elements[0].properties]
    assert len(properties) == 19
    assert set(["x", "y", "z", "red", "green", "blue", "intensity", "classification", "point_source_id"]) <= set(properties)
    # assert colours within 0 - 255
    for col in ["red", "green", "blue"]:
        assert plydata.elements[0].data[col].min() >= 0 and plydata.elements[0].data[col].max() <= 255
    # assert vertex coordinates for first voxel
    voxel1 = dh.df.iloc[0]
    voxel_coords = voxel1[["X", "Y", "Z"]]
    expected_vertex_coords_voxel1 = np.array([
        [voxel_coords["X"], voxel_coords["Y"], voxel_coords["Z"]],
        [voxel_coords["X"] + voxel_size, voxel_coords["Y"], voxel_coords["Z"]],
        [voxel_coords["X"] + voxel_size, voxel_coords["Y"] + voxel_size, voxel_coords["Z"]],
        [voxel_coords["X"], voxel_coords["Y"] + voxel_size, voxel_coords["Z"]],
        [voxel_coords["X"], voxel_coords["Y"], voxel_coords["Z"] + voxel_size],
        [voxel_coords["X"] + voxel_size, voxel_coords["Y"], voxel_coords["Z"] + voxel_size],
        [voxel_coords["X"] + voxel_size, voxel_coords["Y"] + voxel_size, voxel_coords["Z"] + voxel_size],
        [voxel_coords["X"], voxel_coords["Y"] + voxel_size, voxel_coords["Z"] + voxel_size]
    ]) - voxel_size / 2
    # shift to voxel center if requested
    if shift_to_voxel_center:
        all_coords = dh.df[["X", "Y", "Z"]].values
        mins = all_coords.min(axis=0)
        maxs = all_coords.max(axis=0)
        mean_coords = (mins + maxs) / 2
        expected_vertex_coords_voxel1 -= mean_coords
    vertex_coords = np.array(plydata.elements[0].data[["x", "y", "z"]].tolist())
    vertex_coords_voxel1 = vertex_coords[:8]
    assert np.allclose(vertex_coords_voxel1, expected_vertex_coords_voxel1)


def test_write_ply_file_overlapping_voxels(tmp_path, test_file_path_2, datahandler_factory):
    output_file = tmp_path / "output.ply"
    voxel_size = 2
    dh = datahandler_factory(test_file_path_2)
    with pytest.warns(UserWarning):
        dh.save_as_ply(outfile=output_file, voxel_size=voxel_size)
    assert output_file.exists()
