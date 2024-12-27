from pathlib import Path
import pytest
import laspy
import numpy as np
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


def test_remove_buffer():
    pass
