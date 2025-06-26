from pathlib import Path
import pytest
import laspy
import plyfile
import vapc
import shutil #Remove later

@pytest.fixture
def test_file():
    return Path(__file__).parent.parent / "test_data" / "vapc_in.laz"


@pytest.fixture
def test_file_2():
    return Path(__file__).parent.parent / "test_data" / "als_multichannel_leg000_points.laz"


@pytest.fixture
def list_of_test_files():
    return [Path(__file__).parent.parent / "test_data" / "vapc_in.laz", 
            Path(__file__).parent.parent / "test_data" / "als_multichannel_leg000_points.laz"]


@pytest.fixture
def vapc_command_compute():
    return {
        "tool": "compute",
        "args": {
            "compute": [
                "point_density",
                "point_count",
                "percentage_occupied"]
                }
    }


def get_expected_las_fields(vapc_command):
    """
    Get the expected fields in the output las file based on the input vapc_command.

    Parameters
    ----------
    vapc_command : dict
        Dictionary containing Vapc command configurations, including the tool to use
        and any additional arguments.

    Returns
    -------
    list
        List of expected fields in the output las file
    """
    attributes = vapc_command["args"]["compute"]
    if "voxel_index" in attributes:
        attributes.remove("voxel_index")
    if "center_of_gravity" in attributes:
        attributes.remove("center_of_gravity")
        attributes += ["cog_x", "cog_y", "cog_z"]
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
            "Eigenentropy",
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
        # not added as field, just printed to the user
        attributes.remove("percentage_occupied")
    return attributes


def test_do_vapc_on_files_filenotfound(tmp_path, vapc_command_compute):
    """
    Test that the `do_vapc_on_files` function raises a FileNotFoundError when the input file does not exist.
    """
    voxel_size = 0.5
    with pytest.raises(FileNotFoundError):
        vapc.do_vapc_on_files(
            file="non_existent_file.laz",
            out_dir=tmp_path,
            voxel_size=voxel_size,
            vapc_command=vapc_command_compute,
        )


def test_do_vapc_on_files_invalid_arg(test_file, tmp_path, vapc_command_compute):
    """
    Test that the `do_vapc_on_files` function raises a TypeError when an invalid argument is passed.
    """
    voxel_size = 0.5
    with pytest.raises(TypeError):
        vapc.do_vapc_on_files(
            file=test_file,
            out_dir=tmp_path,
            voxel_size=voxel_size,
            vapc_command=vapc_command_compute,
            tilesize=20,  # invalid param
            reduce_to="center_of_voxel",
            save_as=".las",
        )


def test_do_vapc_on_files_invalid_tool_name(test_file, tmp_path):
    """
    Test that the `do_vapc_on_files` function raises a ValueError when an invalid tool
    name is passed provided in the vapc_command.
    """
    voxel_size = 0.5
    vapc_command = {
        "tool": "invalid_tool",
        "args": {
            "compute": [
                "point_density"]
                }
    }
    with pytest.raises(ValueError):
        vapc.do_vapc_on_files(
            file=test_file,
            out_dir=tmp_path,
            voxel_size=voxel_size,
            vapc_command=vapc_command
        )


def test_do_vapc_on_files_invalid_computation(test_file, tmp_path):
    """
    Test that the `do_vapc_on_files` function raises a ValueError when an invalid computation
    is requested in the vapc_command.
    """
    voxel_size = 0.5
    vapc_command = {
        "tool": "compute",
        "args": {
            "compute": [
                "median"
            ]
        }
    }
    with pytest.raises(ValueError):
        vapc.do_vapc_on_files(
            file=test_file,
            out_dir=tmp_path,
            voxel_size=voxel_size,
            vapc_command=vapc_command
        )


@pytest.mark.parametrize("voxel_size,vapc_command",
                         [
                             [0.5, {"tool": "compute", "args": {"compute": ["point_density", "point_count"]}}],
                             [0.5, {"tool": "compute", "args": {"compute": ["eigenvalues", "covariance_matrix"]}}],
                             [0.5, {"tool": "compute", "args": {"compute": ["center_of_gravity", "distance_to_center_of_gravity"]}}],
                             [0.5, {"tool": "compute", "args": {"compute": ["eigenvalues", "geometric_features"]}}],
                             [0.5, {"tool": "compute", 
                                    "args": {"compute": 
                                             ["voxel_index",
                                              "center_of_gravity",
                                              "std_of_cog",
                                              "center_of_voxel",
                                              "corner_of_voxel",
                                              "percentage_occupied"
                                              ]}}]
                         ])
def test_do_vapc_on_one_file_defaults(test_file, tmp_path, voxel_size, vapc_command, capfd):
    """
    Test that the `do_vapc_on_files` function processes a single file correctly
    with default values for the optional parameters.
    """
    vapc.do_vapc_on_files(
        file=test_file,
        out_dir=tmp_path,
        voxel_size=voxel_size,
        vapc_command=vapc_command
    )
    assert len(list(tmp_path.glob("*.laz"))) == 1
    outfile_las = list(tmp_path.glob("*.laz"))[0]
    outfile_json = list(tmp_path.glob("*.json"))[0]
    las = laspy.read(outfile_las)
    attributes = get_expected_las_fields(vapc_command)
    if "percentage_occupied" in vapc_command["args"]["compute"]:
        captured = capfd.readouterr()
        assert "13.43 percent of the voxel space is occupied" in captured.out
    for attr in attributes:
        # ignore colums which we know are not numeric (i.e., not in the las file)
        if not attr == "voxel_index":
            assert las.points[attr] is not None



def test_do_vapc_on_files_defaults(list_of_test_files, tmp_path, vapc_command_compute):
    """
    Test that the `do_vapc_on_files` function processes multiple files correctly
    """
    voxel_size = 1.0
    vapc.do_vapc_on_files(
        file=list_of_test_files,
        out_dir=tmp_path,
        voxel_size=voxel_size,
        vapc_command=vapc_command_compute,
    )
    outfiles = list(tmp_path.glob("*.laz"))
    assert len(outfiles) == 1
    outfile_json = list(tmp_path.glob("*.json"))[0]
    assert outfile_json.exists()
    attributes = get_expected_las_fields(vapc_command_compute)
    las = laspy.read(outfiles[0])
    for attr in attributes:
        if not attr == "voxel_index":
            assert las.points[attr] is not None
    # assert number of points
    assert len(las.points) == 13_522



@pytest.mark.parametrize("voxel_size,vapc_command,tile,reduce_to,save_as", 
                         [
                             [1.0, {"tool": "compute",
                                    "args": {"compute":
                                             ["voxel_index",
                                              "center_of_gravity",
                                              ]}},
                                              20,
                                              "center_of_voxel",
                                              ".las"],
                             [1.0, {"tool": "compute",
                                    "args": {"compute":
                                             ["voxel_index",
                                              "center_of_gravity",
                                              ]}},
                                              20,
                                              "center_of_voxel",
                                              ".ply"],
                         ])
def test_do_vapc_on_one_file(test_file_2, tmp_path, voxel_size, vapc_command, tile, reduce_to, save_as):
    """
    Test that the `do_vapc_on_files` function processes a single file correctly
    with non-default values for the optional parameters.
    """
    vapc.do_vapc_on_files(
        file=test_file_2,
        out_dir=tmp_path,
        voxel_size=voxel_size,
        vapc_command=vapc_command,
        tile=tile,
        reduce_to=reduce_to,
        save_as=save_as
    )
    assert len(list(tmp_path.glob(f"*{save_as}"))) == 1
    outfile = list(tmp_path.glob(f"*{save_as}"))[0]
    outfile_json = list(tmp_path.glob("*.json"))[0]
    attributes = get_expected_las_fields(vapc_command)
    if save_as == ".las" or save_as == ".laz":
        las = laspy.read(outfile)
        for attr in attributes:
            # ignore colums which we know are not numeric (i.e., not in the las file)
            if not attr == "voxel_index":
                assert las.points[attr] is not None
    elif save_as == ".ply":
        plydata = plyfile.PlyData.read(outfile)
        #List of properties we can ignore
        properties_to_ignore = ['x', 'y', 'z', 'intensity', 'bit_fields', 'classification_flags', 'classification', 'user_data', 'scan_angle', 'point_source_id', 'gps_time', 'red', 'green', 'blue', 'echo_width', 'fullwaveIndex', 'hitObjectId', 'heliosAmplitude','return_number', 'number_of_returns', 'synthetic', 'key_point', 'withheld', 'overlap', 'scanner_channel', 'scan_direction_flag', 'edge_of_flight_line',]
        properties = [p.name for p in plydata.elements[0].properties if p.name not in properties_to_ignore]
        print(properties)
        print(attributes)
        assert len(properties) == 3
        assert set(attributes) <= set(properties)
