import laspy
from laspy import LaspyException
import numpy as np
import os
from pathlib import Path
from functools import partial


def las_create_or_append(fh, las, mask, tile_name):
    """
    Creates a new LAS file or appends points to an existing one based on the provided mask.

    Parameters:
    - fh (laspy.LasFile): The LAS file handle containing header information.
    - las (laspy.LasData): The LAS data object containing point data.
    - mask (numpy.ndarray): A boolean mask to filter points to be written.
    - tile_name (str): The file path for the tile to create or append to.

    Returns:
    - str or bool: Returns True if successful, or False if no points to write.
    """
    if os.path.exists(tile_name):
        # If the outfile exists, append points to it
        with laspy.open(tile_name, "a") as lf:
            app_p = las.points[mask]
            print(app_p)
            lf.append_points(app_p)
            print("Appended points to:\t", tile_name)
    else:
        header = laspy.LasHeader(
            point_format=fh.header.point_format, version=fh.header.version
        )
        header.offsets = fh.header.offsets
        header.scales = fh.header.scales
        las_tile = laspy.LasData(header)
        if las.points.x[mask].shape[0] == 0:
            return False
        las_tile.points = las.points[mask]
        if "HELIOS++" in las.header.generating_software or "HELIOS++" in header.generating_software:
            try:
                las_tile.remove_extra_dims(["ExtraBytes"])
            except LaspyException:
                pass  # 'ExtraBytes' dimension does not exist
        las_tile.write(tile_name)
        print("Created file:\t\t", tile_name)
    return True


def las_merge(filepaths, outfile):
    """
    Merges multiple LAS files into a single output file.

    Parameters:
    - filepaths (list of str): List of file paths to LAS files to be merged.
    - outfile (str): The file path for the merged output LAS file.

    Returns:
    - True
    """
    for i, f in enumerate(filepaths):
        with laspy.open(f, "r") as lf_0:
            las = lf_0.read()
            if len(las.points) == 0:
                print("Empty file, skipping...")
                continue
            else:
                las.write(outfile)
                print(f"Created {outfile}. Merging now...")
                break
    with laspy.open(outfile, "a") as lf:
        scales = lf.header.scales
        offsets = lf.header.offsets
        for file in filepaths[1:]:
            with laspy.open(file) as lf_a:
                lf_aa = lf_a.read()
                lf_aa.X = (lf_aa.x - offsets[0]) / scales[0]
                lf_aa.Y = (lf_aa.y - offsets[1]) / scales[1]
                lf_aa.Z = (lf_aa.z - offsets[2]) / scales[2]
                lf.append_points(lf_aa.points)
    return True


def lasz_to_lasz(infile, outfile=False):
    """
    Converts a LAS file to a LAZ file or a LAZ file to a LAS file.

    Parameters:
    - infile (str): The input LAS file path.
    - outfile (str): The output LAZ file path.

    Returns:
    - None
    """
    infile = str(infile)
    if outfile is False:
        if infile[-1] == "z":
            outfile = infile[:-4] + ".las"
        elif infile[-1] == "s":
            outfile = infile[:-4] + ".laz"
        else:
            return False
    with laspy.open(infile) as lasf:
        las = lasf.read()
        las.write(outfile)


def las_create_3dtiles(lazfile, out_dir, tilesize, tilename="", buffer=0):
    """
    Creates 3D tiles from a LAZ file by partitioning the point cloud into tiles of specified size.

    Parameters:
    - lazfile (str): Path to the input LAZ file.
    - out_dir (str): Directory where the output tile files will be stored.
    - tilesize (float): The size of each tile in the X, Y, and Z directions.
    - tilename (str, optional): An optional prefix for the tile file names. Defaults to "".
    - buffer (float, optional): Buffer size to expand each tile boundary. Defaults to 0.

    Returns:
    - numpy.ndarray: Array of unique tile file paths created.
    """
    assert isinstance(tilesize, (int, float)), 'tilesize must be a number'
    assert isinstance(buffer, (int, float)), 'buffer must be a number'
    assert tilesize > 0, "Tile size must be greater than 0"
    assert buffer >= 0, "Buffer size must be greater than or equal to 0"
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    with laspy.open(lazfile, "r") as fh:
        las = fh.read()
        extent = [*fh.header.min, *fh.header.max]
        x_min = np.floor(extent[0])
        x_max = np.ceil(extent[3])
        y_min = np.floor(extent[1])
        y_max = np.ceil(extent[4])
        z_min = np.floor(extent[2])
        z_max = np.ceil(extent[5])
        if (
            x_min == 0
            and x_max == 0
            and y_min == 0
            and y_max == 0
            and z_min == 0
            and z_max == 0
        ):
            x_min, y_min, z_min = las.xyz.min(axis=0)
            x_max, y_max, z_max = las.xyz.max(axis=0)
            # assert False, "extent not defined in header of las/laz file"
    # Find number of tiles for x and y direction
    diff_x = x_max - x_min
    diff_y = y_max - y_min
    diff_z = z_max - z_min
    x_tiles = int(np.ceil(diff_x / tilesize))
    y_tiles = int(np.ceil(diff_y / tilesize))
    z_tiles = int(np.ceil(diff_z / tilesize))

    # Define x,y boundaries for the tiles
    x_verts = np.arange(x_min, x_min + x_tiles * tilesize + 1, tilesize)
    y_verts = np.arange(y_min, y_min + y_tiles * tilesize + 1, tilesize)
    z_verts = np.arange(z_min, z_min + z_tiles * tilesize + 1, tilesize)
    tile_names = []
    masks = []
    print(f"Generating masks for {x_tiles * y_tiles * z_tiles} potential tiles ... ")
    for x_tile in range(x_tiles):
        tile_min_x = round(x_verts[x_tile], 3)
        tile_max_x = round(x_verts[x_tile + 1], 3)
        for y_tile in range(y_tiles):
            tile_min_y = round(y_verts[y_tile], 3)
            tile_max_y = round(y_verts[y_tile + 1], 3)
            for z_tile in range(z_tiles):
                tile_min_z = round(z_verts[z_tile], 3)
                tile_max_z = round(z_verts[z_tile + 1], 3)
                out_tile = os.path.join(
                    out_dir,
                    f"{tilename}_{tile_min_x}_{tile_min_y}_{tile_min_z}_{tilesize}_tileSize_{buffer}_buff.las",
                )
                m_min_x = las.points.x >= (tile_min_x - buffer)
                m_max_x = las.points.x < (tile_max_x + buffer)
                m_min_y = las.points.y >= (tile_min_y - buffer)
                m_max_y = las.points.y < (tile_max_y + buffer)
                m_min_z = las.points.z >= (tile_min_z - buffer)
                m_max_z = las.points.z < (tile_max_z + buffer)
                mask_all = m_min_x & m_max_x & m_min_y & m_max_y & m_min_z & m_max_z
                # outfile_mask_dir[outTile] = mask_all

                tile_names.append(out_tile)
                masks.append(mask_all)
    print("Created masks")
    write_tile_for_lasfile = partial(las_create_or_append, fh, las)

    # Clip point clouds
    existing_tiles = []
    for mask, tile_name in zip(masks, tile_names):
        existing_tiles.append(write_tile_for_lasfile(mask, tile_name))
    return np.array(tile_names)[existing_tiles]


def clip_to_bbox(laz_in, laz_out, bbox):
    """
    Clips a LAZ file to a specified bounding box and writes the result to a new file.

    Parameters:
    - laz_in (str): Path to the input LAZ file.
    - laz_out (str): Path for the output clipped LAZ file.
    - bbox (list or tuple of float): Bounding box defined as [x_min, y_min, z_min, x_max, y_max, z_max].

    Returns:
    - int: The number of points in the clipped file, or 0 if no points remain.
    """
    with laspy.open(laz_in, "r") as fh:
        las = fh.read()

    xmi = las.points.x >= bbox[0]
    xma = las.points.x < bbox[3]

    ymi = las.points.y >= bbox[1]
    yma = las.points.y < bbox[4]

    zmi = las.points.z >= bbox[2]
    zma = las.points.z < bbox[5]
    points_masked = las.points[xmi & xma & ymi & yma & zmi & zma]

    if points_masked.X.shape[0] == 0:
        print(f"File empty after removing buffer: {laz_in}")
        os.remove(laz_in)
        return 0
    ct = points_masked.X.shape[0]
    header = laspy.LasHeader(
        point_format=fh.header.point_format, version=fh.header.version
    )
    header.offsets = fh.header.offsets
    header.scales = fh.header.scales

    lastile = laspy.LasData(header)
    lastile.points = points_masked
    lastile.write(laz_out)
    return ct


def las_remove_buffer(folder):
    """
    Removes buffer zones from all LAS files in a specified folder by clipping them to their bounding boxes.
    The bounding box is encoded in the file name (min coords + tile size).

    Parameters:
    - folder (str): Path to the folder containing LAS files to process.

    Returns:
    - list of str: List of output LAS files after buffer removal.
    """
    ofs = []
    for file in os.listdir(str(folder)):
        if not file.endswith(".las") and not file.endswith(".laz"):
            continue
        bbox_str = file.split("_")[1:4]
        tilesize = float(file.split("_")[4].split(".")[0])
        x_min, y_min, z_min, x_max, y_max, z_max = (
            float(bbox_str[0]),
            float(bbox_str[1]),
            float(bbox_str[2]),
            float(bbox_str[0]) + tilesize,
            float(bbox_str[1]) + tilesize,
            float(bbox_str[2]) + tilesize,
        )
        bbox = np.round([x_min, y_min, z_min, x_max, y_max, z_max], 4)
        in_file = os.path.join(folder, file)
        out_file = in_file
        clip_to_bbox(in_file, out_file, bbox)
        if os.path.isfile(out_file):
            ofs.append(out_file)
    return ofs
