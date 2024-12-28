from pathlib import Path
import pytest
from unittest.mock import patch, call
import laspy
import numpy as np
import vapc
from vapc import las_split_append_merge
# from vapc.las_split_append_merge import clip_to_bbox


@pytest.fixture
def test_file_path_1():
    return Path(__file__).parent.parent / "test_data" / "vapc_in.laz"


@pytest.fixture
def test_file_path_2():
    return Path(__file__).parent.parent / "test_data" / "small_mask.laz"


@pytest.fixture
def test_file_path_3():
    return Path(__file__).parent.parent / "test_data" / "ALS_BR04_2019-07-05_140m.laz"


@pytest.fixture
def tiling_test_input(tmp_path):
    minx1, miny1, minz1, tilesize1 = (0, 0, 0, 10)
    minx2, miny2, minz2, tilesize2 = (476850, 5429100, 200, 100)
    minx3, miny3, minz3, tilesize3 = (-10, -10, 0, 20)
    bbox1 = [minx1, miny1, minz1, minx1 + tilesize1, miny1 + tilesize1, minz1 + tilesize1]
    bbox2 = [minx2, miny2, minz2, minx2 + tilesize2, miny2 + tilesize2, minz2 + tilesize2]
    bbox3 = [minx3, miny3, minz3, minx3 + tilesize3, miny3 + tilesize3, minz3 + tilesize3]
    # create files in the folder
    file1 = tmp_path / f"area_{minx1}_{miny1}_{minz1}_{tilesize1}.laz"
    file2 = tmp_path / f"area_{minx2}_{miny2}_{minz2}_{tilesize2}.laz"
    file3 = tmp_path / f"area_{minx3}_{miny3}_{minz3}_{tilesize3}.laz"
    for f in [file1, file2, file3]:
        f.touch()
    return tmp_path, [str(file1), bbox1], [str(file2), bbox2], [str(file3), bbox3]


def test_create_las():
    pass


def test_append_to_las():
    pass


@pytest.mark.parametrize("file_ext", ["las", "laz"])
def test_merge_las(tmp_path, file_ext, test_file_path_1, test_file_path_2):
    filepaths = [test_file_path_1, test_file_path_2]
    outfile = tmp_path / f"merged.{file_ext}"
    las_split_append_merge.las_merge(filepaths, outfile)
    assert outfile.exists()
    n_pts = 0
    for f in filepaths:
        las = laspy.read(f)
        n_pts += len(las.points)
    las_merged = laspy.read(outfile)
    assert len(las_merged.points) == n_pts


def test_compress_las(test_file_path_1):
    outfile = Path(str(test_file_path_1).replace(".laz", ".las"))
    # test decompress
    las_split_append_merge.laSZ_to_laSZ(test_file_path_1)
    assert outfile.exists()
    las_in = laspy.read(test_file_path_1)
    las_out = laspy.read(outfile)
    assert las_in.points == las_out.points
    # test compress
    outfile2 = Path(str(outfile).replace(".las", "_2.laz"))
    las_split_append_merge.laSZ_to_laSZ(outfile, outfile2)
    assert outfile2.exists()
    las_out2 = laspy.read(outfile2)
    assert las_in.points == las_out2.points
    # clean up
    outfile.unlink()
    outfile2.unlink()


def test_tile_las():
    pass


def test_clip_las(test_file_path_3, tmp_path):
    bbox = [476900, 5429150, 240, 476950, 5429200, 340]
    outfile = tmp_path / "clipped.laz"
    n_pts_written = las_split_append_merge.clip_to_bbox(test_file_path_3, outfile, bbox)
    print(n_pts_written)
    assert outfile.exists()
    las = laspy.read(outfile)
    assert las.header.min[0] >= bbox[0]
    assert las.header.min[1] >= bbox[1]
    assert las.header.min[2] >= bbox[2]
    assert las.header.max[0] <= bbox[3]
    assert las.header.max[1] <= bbox[4]
    assert las.header.max[2] <= bbox[5]
    assert min(las.x) >= bbox[0]
    assert min(las.y) >= bbox[1]
    assert min(las.z) >= bbox[2]
    assert max(las.x) <= bbox[3]
    assert max(las.y) <= bbox[4]
    assert max(las.z) <= bbox[5]


def test_remove_buffer(tiling_test_input):
    # mock the clip_to_bbox function
    with patch('vapc.las_split_append_merge.clip_to_bbox') as mock_clip_to_bbox:
        # Call the function_to_test
        outfiles = vapc.las_split_append_merge.las_remove_buffer(tiling_test_input[0])
        # Assert clip_to_bbox was called as often as expected
        assert mock_clip_to_bbox.call_count == len(tiling_test_input[1:])
        # get calls of clip_to_bbox
        call_args = list(mock_clip_to_bbox.call_args_list)
        match = 0
        for (file, expected_bbox) in tiling_test_input[1:]:
            # not possible this way - fails at array comparison - we need to use numpy
            # mock_clip_to_bbox.assert_any_call(file, file, expected_bbox)
            assert file in outfiles
            expected_bbox = [float(el) for el in expected_bbox]
            # loop since calls may not be in the same order
            for func_call in call_args:
                (infile, outfile, bbox) = func_call.args
                if infile == file:
                    assert outfile == file
                    np.testing.assert_array_equal(bbox, expected_bbox)
                    match += 1
                    break
        assert match == len(outfiles)
