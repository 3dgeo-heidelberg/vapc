from pathlib import Path
import pytest
import vapc


@pytest.fixture
def test_file():
    return Path(__file__).parent.parent / "test_data" / "vapc_in.laz"


@pytest.fixture
def list_of_test_files():
    return ["../tests/test_data/vapc_in.laz", "../tests/test_data/vapc_in.laz"]


@pytest.fixture
def vapc_command_compute():
    return {
        "tool": "compute",
        "args": {
            "compute": [
                "point_density",
                "point_count",
                "percentage_occupied",
                "geometric_features",
                "center_of_gravity",
                "center_of_voxel"]
                }
    }

def test_do_vapc_on_files_filenotfound(tmp_path, vapc_command_compute):
    voxel_size = 0.5
    with pytest.raises(FileNotFoundError):
        vapc.do_vapc_on_files(
            file="non_existent_file.laz",
            out_dir=tmp_path,
            voxel_size=voxel_size,
            vapc_command=vapc_command_compute,
        )


def test_do_vapc_on_files_invalid_arg(test_file, tmp_path, vapc_command_compute):
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


@pytest.mark.parametrize("voxel_size,vapc_command",
                         [
                             [0.5, {"tool": "compute", "args": {"compute": ["point_density", "point_count"]}}],
                         ])
def test_do_vapc_on_one_file_defaults(test_file, tmp_path, voxel_size, vapc_command):
    pass


@pytest.mark.parametrize("voxel_size,vapc_command",
                         [
                             [0.5, {"tool": "compute", "args": {"compute": ["point_density", "point_count"]}}],
                         ])
def test_do_vapc_on_files_defaults(list_of_test_files, tmp_path, voxel_size, vapc_command):
    pass


@pytest.mark.parametrize("voxel_size,vapc_command,tile,reduce_to,save_as", 
                         [
                             [0.5, {"tool": "compute", "args": {"compute": ["point_density", "point_count"]}}, 20, "center_of_voxel", ".las"],
                         ])
def test_do_vapc_on_one_file(test_file, tmp_path, voxel_size, vapc_command, tile, reduce_to, save_as):
    pass


# TODO: Test all jsons in "command_line_templates" folder