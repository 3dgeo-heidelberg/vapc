from pathlib import Path
import pytest
from unittest.mock import patch, call
import laspy
import numpy as np
import vapc
from vapc import las_split_append_merge


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
    """
    Fixture to create test input for the las_remove_buffer function (mocking the clip_to_bbox function)
    """
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


@pytest.mark.parametrize("file_ext", ["las", "laz"])
def test_merge_las(tmp_path, file_ext, test_file_path_1, test_file_path_2):
    """
    Test merging two LAS/LAZ files
    """
    filepaths = [test_file_path_1, test_file_path_2]
    outfile = tmp_path / f"merged.{file_ext}"
    las_split_append_merge.las_merge(filepaths, outfile)
    # check if the file exists
    assert outfile.exists()
    # get expected number of points (sum of points in input files)
    n_pts = 0
    for f in filepaths:
        las = laspy.read(f)
        n_pts += len(las.points)
    las_merged = laspy.read(outfile)
    # check if the number of points in the merged file is correct
    assert len(las_merged.points) == n_pts


def test_compress_las(test_file_path_1):
    """
    Test compressing and decompressing a LAS/LAZ file
    """
    outfile = Path(str(test_file_path_1).replace(".laz", ".las"))
    # test decompress
    las_split_append_merge.lasz_to_lasz(test_file_path_1)
    assert outfile.exists()
    las_in = laspy.read(test_file_path_1)
    las_out = laspy.read(outfile)
    assert las_in.points == las_out.points
    # test compress
    outfile2 = Path(str(outfile).replace(".las", "_2.laz"))
    las_split_append_merge.lasz_to_lasz(outfile, outfile2)
    assert outfile2.exists()
    las_out2 = laspy.read(outfile2)
    assert las_in.points == las_out2.points
    # clean up
    outfile.unlink()
    outfile2.unlink()


@pytest.mark.parametrize("tilesize, expected_num_tiles",
                         [[50.0, 27]])
def test_tile_las(tmp_path, test_file_path_3, tilesize, expected_num_tiles):
    """
    Test creating 3D tiles from a LAS file
    """
    outfolder = tmp_path / "tiles"
    outfiles = las_split_append_merge.las_create_3dtiles(test_file_path_3, outfolder, tilesize)
    # get files
    found_files = list(Path(outfolder).glob("*.las"))
    assert len(found_files) == len(outfiles)
    assert len(outfiles) == expected_num_tiles


@pytest.mark.parametrize("tilesize, buffersize",
                         [[-50.0, 0],
                          [0.0, 0],
                          ["50", 0],
                          [50, -10],
                          [50, "ten"]])
def test_tile_las_invalid_tile_buffer_size(tmp_path, test_file_path_3, tilesize, buffersize):
    """
    Test that an error is raised when providing invalid tilesize or buffer size
    to las_create_3dtiles function
    """
    outfolder = tmp_path / "tiles"
    with pytest.raises(AssertionError):
        las_split_append_merge.las_create_3dtiles(test_file_path_3,
                                                  outfolder,
                                                  tilesize,
                                                  buffer=buffersize)


def test_clip_las(test_file_path_3, tmp_path):
    """
    Test clipping a LAS file to a bounding box
    """
    # bounding box for clipping
    bbox = [476900, 5429150, 240, 476950, 5429200, 340]
    outfile = tmp_path / "clipped.laz"
    n_pts_written = las_split_append_merge.clip_to_bbox(test_file_path_3, outfile, bbox)
    # check if the file exists after executing the function
    assert outfile.exists()
    # read output file and check if the bounding box is as expected
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
    """
    Test removing buffer zones from LAS files in a folder with las_remove_buffer function
    """
    # mock the clip_to_bbox function
    with patch('vapc.las_split_append_merge.clip_to_bbox') as mock_clip_to_bbox:
        # Call the function to test
        outfiles = vapc.las_split_append_merge.las_remove_buffer(tiling_test_input[0])
        # Assert clip_to_bbox was called as often as expected (once for each file in the folder)
        assert mock_clip_to_bbox.call_count == len(tiling_test_input[1:])
        # get calls of clip_to_bbox (with args) to check if the function was called with the correct arguments
        call_args = list(mock_clip_to_bbox.call_args_list)
        match = 0
        for (file, expected_bbox) in tiling_test_input[1:]:
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
        # check if all files were found
        assert match == len(outfiles)
